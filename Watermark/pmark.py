# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# Pmark.py
# Description: Implementation of Pmark algorithm
# ============================================

import torch
import numpy as np
from sympy.codegen.ast import continue_
#  from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import PreTrainedTokenizer
from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from sentence_transformers import SentenceTransformer, models
from nearpy.hashes import RandomBinaryProjections
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Callable, Iterator
from typing import List, Dict, Any
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import re
import math
import random
from scipy.stats import beta
from nltk.tokenize import sent_tokenize
import nltk
from visualize.data_for_visualization import DataForVisualization

message = '001011111000100110110'
global watermark_count


class PmarkConfig(BaseConfig):
    """Config class for Pmark algorithm.load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.max_new_tokens = self.config_dict['max_new_tokens']
        self.min_new_tokens = self.config_dict['min_new_tokens']
        self.path_to_embedder = self.config_dict['path_to_embedder']
        self.N_max = self.config_dict['N_max']
        self.gamma = self.config_dict['gamma']
        self.margin_m = self.config_dict['margin_m']
        self.dimension_d = self.config_dict['dimension_d']
        self.prime_P = self.config_dict['prime_P']
        self.threshold = self.config_dict['threshold']
        self.path_to_centroids = self.config_dict['path_to_centroids']
        # 分段数

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "Pmark"


# utils


class PmarkUtils:
    """Helper class for Pmark algorithm, contains helper functions."""

    def __init__(self, config: PmarkConfig, *args, **kwargs) -> None:
        """
            Initialize the Pmark utility class.

            Parameters:
                config (PmarkConfig): Configuration for the Pmark algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    class SBERTLSHModel:
        """Helper class for SBERTLSHModel"""

        def __init__(self, batch_size, lsh_dim, sbert_type='roberta', lsh_model_path=None, **kwargs):
            self.comparator: Callable[[np.ndarray, np.ndarray], float]
            self.do_lsh: bool = False
            self.dimension: int = -1
            self.batch_size: int = batch_size
            self.lsh_dim: int = lsh_dim
            print("initializing random projection LSH model")
            self.hasher = RandomBinaryProjections(
                'rbp_perm', projection_count=self.lsh_dim, rand_seed=1234)
            self.do_lsh = True
            self.comparator = lambda x, y: cosine(x, y)
            self.sbert_type = sbert_type
            self.dimension = 1024 if 'large' in self.sbert_type else 768
            # print(f'loading SBERT {self.sbert_type} model...')
            if lsh_model_path is not None:
                # lsh_model_path = r"E:\models\all-mpnet-base-v2\models--sentence-transformers--all-mpnet-base-v2\snapshots\e8c3b32edf5434bc2275fc9bab85f82640a19130"
                word_embedding_model = models.Transformer(lsh_model_path)
                pooling_model = models.Pooling(
                    word_embedding_model.get_word_embedding_dimension(),
                    pooling_mode_mean_tokens=True
                )
                self.embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                self.dimension = self.embedder.get_sentence_embedding_dimension()
            else:
                embedding_dir = r"E:\models\all-mpnet-base-v2\models--sentence-transformers--all-mpnet-base-v2\snapshots\e8c3b32edf5434bc2275fc9bab85f82640a19130"
                self.embedder = SentenceTransformer(
                    embedding_dir)
            self.embedder.eval()

            self.hasher.reset(dim=self.dimension)

        def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
            all_embeddings = self.embedder.encode(sents, batch_size=self.batch_size)
            return np.stack(all_embeddings)

        def get_hash(self, sents: Iterator[str]) -> Iterator[str]:
            embd = self.get_embeddings(sents)
            hash_strs = [self.hasher.hash_vector(e)[0] for e in embd]
            hash_ints = [int(s, 2) for s in hash_strs]
            return hash_ints

    @staticmethod
    def pairwise_cosine(data1, data2, device=torch.device('cpu')):
        data1, data2 = data1.to(device), data2.to(device)

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
        A_normalized = A / A.norm(dim=-1, keepdim=True)
        B_normalized = B / B.norm(dim=-1, keepdim=True)

        cosine = A_normalized * B_normalized

        # return N*N matrix for pairwise distance
        cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
        return cosine_dis

    class SentenceEndCriteria(StoppingCriteria):
        """
        ONLY WORK WITH BATCH SIZE 1

        Stop generation whenever the generated string is **more than one** sentence (i.e. one full sentence + one extra token). this is determined by nltk sent_tokenize.
        Only stop if ALL sentences in the batch is at least two sentences

        Args:
            tokenizer (PreTrainedTokenizer):
            The exact tokenizer used for generation. MUST BE THE SAME!
        """

        def __init__(self, tokenizer: PreTrainedTokenizer):
            self.tokenizer = tokenizer
            self.current_num_sentences = 0

        def update(self, current_text):
            self.current_num_sentences = len(sent_tokenize(current_text))

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            assert input_ids.size(0) == 1

            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            k = sent_tokenize(text)

            return len(k) > self.current_num_sentences + 1
            # return len(sent_tokenize(text)) > self.current_num_sentences + 1


# main class


class Pmark(BaseWatermark):
    """Top-level class for the Pmark algorithm."""

    def __init__(self, algorithm_config: str | PmarkConfig, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        """
            Initialize the Pmark algorithm.

            Parameters:
                algorithm_config (str | PmarkConfig): Path to the algorithm configuration file or PmarkConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = PmarkConfig(
                algorithm_config, transformers_config)
        elif isinstance(algorithm_config, PmarkConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or a PmarkConfig instance")

        self.utils = PmarkUtils(self.config)

        self.segments_d = []
        self.text_ids = []

        self.segment_usage_counts = {}

        self.segment_bag = []
        self.collected_features: List[Dict[str, Any]] = []

        self.shared_embedder = self.utils.SBERTLSHModel(
            lsh_model_path=self.config.path_to_embedder, batch_size=1, lsh_dim=3,
            sbert_type='base'
        )

        # self.anchor_generator = RobustAnchorGenerator(
        #     mode='cluster',
        #     embedder=self.shared_embedder,
        #     device=self.config.device
        # )

        self.anchor_generator = RobustAnchorGenerator(
            mode='topk_sign',  # <--- 将默认模式改为 SSB
            embedder=self.shared_embedder,
            device=self.config.device,
            top_k=64  # <--- 增加提取 Top-K 个显著维度
        )

    def _prepare_fixed_segments(self, message: str) -> List[str]:
        """
        将消息分割为固定长度的段 （改进：基于参考句子的语义复杂度动态分段， # 如果没有参考句子，使用固定分段）

        Args:
            message: 二进制消息
            segment_length: 每段比特数，如果为None则自动计算

        Returns:
            segments: 固定长度的段列表
        """
        # 确定分段长度
        # if hasattr(self, 'bits_per_segment'):
        #     segment_length = self.bits_per_segment
        # else:
        # 默认：根据总比特数和期望的句子数计算
        target_sentences = getattr(self, 'target_sentences', 7)
        segment_length = max(1, math.ceil(len(message) / target_sentences))

        # 确保段长度合理
        segment_length = max(1, min(segment_length, len(message)))
        # 分段
        segments = []
        for i in range(0, len(message), segment_length):
            segment = message[i:i + segment_length]

            # 填充最后一段使其达到固定长度
            if len(segment) < segment_length:
                # 方法1：填充0
                segment = segment.ljust(segment_length, '0')
                # 方法2：或者可以添加特殊的结束标记
                # segment = segment + '1' + '0' * (segment_length - len(segment) - 1)

            segments.append(segment)
        lenth = len(segments[0])

        return segments, lenth

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the Pmark algorithm with Residual/Anchor Logic."""
        segments, lenth = self._prepare_fixed_segments(message)
        lsh_model = self.utils.SBERTLSHModel(lsh_model_path=self.config.path_to_embedder, batch_size=1, lsh_dim=lenth,
                                             sbert_type='base')
        normals = torch.tensor(lsh_model.hasher.normals, device=self.config.device)

        # cluster_centers = torch.load(self.config.path_to_centroids)

        self.segments_d.clear()
        self.segment_usage_counts.clear()

        L = len(segments[0])
        target_normals = normals[:L]

        self.segments = segments
        N = len(segments)

        # 初始化计数器
        for i in range(N):
            self.segment_usage_counts[i] = 0

        sent_end_criteria = PmarkUtils.SentenceEndCriteria(self.config.generation_tokenizer)

        text = prompt
        text_ids = self.config.generation_tokenizer.encode(prompt, return_tensors='pt').to(self.config.device)

        self.text_ids.append(text_ids)
        prompt_length = len(text_ids[0])
        sent_end_criteria.update(text)  # update should take current text
        current_trials = 0
        window_size = 1
        # new_text_prev = prompt

        while True:
            # --- Shuffle Bag ---
            if len(self.segment_bag) == 0:
                secret_string = 'The quick brown fox jumps over the lazy dog'
                segment_hash = lsh_model.get_hash([secret_string])[0]

                rng = random.Random(segment_hash)
                self.segment_bag = list(range(len(segments))).copy()
                rng.shuffle(self.segment_bag)

            final_segment_index = self.segment_bag.pop()
            segment_to_embed = segments[final_segment_index]  # e.g., [0, 1, 1]
            self.segment_usage_counts[final_segment_index] += 1
            self.segments_d.append(segment_to_embed)

            print(f"生成第{current_trials + 1}句，目标Bit: {segment_to_embed}")
            has_printed = False
            has_printeds = False
            while True:
                stopping_criteria = StoppingCriteriaList([sent_end_criteria])
                outputs = self.config.generation_model.generate(text_ids,
                                                                max_new_tokens=self.config.max_new_tokens,
                                                                min_new_tokens=self.config.min_new_tokens,
                                                                do_sample=True,
                                                                temperature=0.88,
                                                                top_k=0,
                                                                repetition_penalty=1.05,
                                                                stopping_criteria=stopping_criteria,
                                                                )
                new_text_ids = outputs[:, :-1]
                new_text = self.config.generation_tokenizer.decode(
                    new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)

                # 获取候选句 Embedding
                # shape: (num_candidates, embed_dim)
                embeds = lsh_model.get_embeddings(new_text)
                embeds = torch.tensor(embeds, device=self.config.device)

                sentences_new = sent_tokenize(text)

                if current_trials == 0 and len(sent_tokenize(prompt)) != 1:
                    context_window = [prompt]
                    if not has_printeds:
                        print("prompt分句不为1")
                        has_printeds = True
                else:
                    context_window = sentences_new[-window_size:]

                if not has_printed:
                    print("依靠的这一句生成的：", context_window)
                    has_printed = True

                anchor = self.anchor_generator.get_anchor(context_window, self.config.path_to_centroids)  # 语义方向

                current_target_normals = self.orthogonalize_normals(target_normals, anchor)  # 正交 把和语义方向不同的部分分离出来

                # (num_candidates, embed_dim) @ (L, embed_dim).T -> (num_candidates, L)
                projections = torch.matmul(embeds.float(), current_target_normals.T.float())

                # 假设 segment_to_embed 是 [0, 1, 1]，转换成 [-1, 1, 1]
                target_signs = torch.tensor(
                    [1 if str(b) == '1' else -1 for b in segment_to_embed],
                    device=self.config.device,
                    dtype=torch.float
                )

                score = torch.dot(projections, target_signs).item()
                pred_signs = torch.sign(projections)
                current_accuracy = (pred_signs == target_signs).sum().item()
                score_abs = torch.abs(torch.tensor(score))

                if (current_accuracy == len(segment_to_embed) and abs(score_abs) >= self.config.margin_m):
                    current_trials += 1
                    new_text_prev = new_text
                    break
                else:
                    continue

            [new_text] = np.array([new_text])
            accepted_text = list(new_text)
            if (len(accepted_text) == 0 and current_trials < self.config.N_max):
                continue

            text += new_text
            # text = "\n".join([text, new_text])
            text_ids = new_text_ids.to(self.config.device)
            sent_end_criteria.update(text)
            if (len(text_ids[0]) - prompt_length) >= self.config.max_new_tokens - 1:
                break
        watermarked_text = text.strip()
        return watermarked_text

    def _prompt_is_incomplete(self, prompt: str) -> bool:
        prompt = prompt.strip()
        return len(prompt) > 0 and not prompt.endswith(('.', '!', '?'))

    def detect_watermark(self, orign_text: str, text: str, prompt: str, return_dict: bool = True, *args, **kwargs):
        sentences = sent_tokenize(text)
        prompt_is_incomplete = self._prompt_is_incomplete(orign_text[:len(prompt.rstrip('\n')) + 1])
        print(orign_text[:len(prompt) + 1])

        if not prompt_is_incomplete:
            print("prmpt 完整")
            pass
        else:
            if len(sentences) >= 2 and prompt_is_incomplete:
                if len(sent_tokenize(sentences[0])) > 1:
                    sentences[0] = sent_tokenize(sentences[0])[0]
                    remaining_part = " ".join(sent_tokenize(sentences[0])[1:])
                    sentences.insert(1, remaining_part)
                print("扔的句子是：", sentences[0])
                sentences.pop(0)

        L = self.config.dimension_d

        lsh_model = self.utils.SBERTLSHModel(lsh_model_path=self.config.path_to_embedder, batch_size=1, lsh_dim=L,
                                             sbert_type='base')

        normals = torch.tensor(lsh_model.hasher.normals, device=self.config.device).float()
        target_normals = normals[:L]

        detected_segments = []
        raw_scores = []

        current_context = sentences[0]

        total_bits_count = 0
        correct_bits_count = 0

        # print(f"检测开始，共处理 {len(detected_segments)} 个句子")
        if len(sentences) != len(self.segments_d):
            for i in range(len(sentences)):
                print(f"第{i + 1}句为：", sentences[i])

        for i in range(0, len(self.segments_d)):
            target_sentence = sentences[i]

            if i == 0:
                context_window = [prompt]

            else:
                start_idx = max(0, i - 1)
                context_window = sentences[start_idx:i]

            print("依靠的这一句检测的：", context_window)

            anchor = self.anchor_generator.get_anchor(context_window, self.config.path_to_centroids)

            current_target_normals = self.orthogonalize_normals(target_normals, anchor)

            # --- B. 获取当前句向量 ---
            target_numpy = lsh_model.get_embeddings([target_sentence])
            target_embed = torch.tensor(target_numpy, device=self.config.device).float()  # (1, dim)

            scores = torch.matmul(target_embed.float(), current_target_normals.T.float())

            # 提取 bits: score > 0 为 1, score < 0 为 0
            bits = (scores > 0).int().tolist()[0]  # e.g. [1, 0, 1]

            detected_segments.append(bits)
            raw_scores.append(scores.tolist()[0])

            current_context += target_sentence

        count = 0

        for i in range(len(detected_segments)):
            det_str = "".join(map(str, detected_segments[i]))

            if isinstance(self.segments_d[i], list):
                truth_str = "".join(map(str, self.segments_d[i]))
            else:
                truth_str = str(self.segments_d[i])

            if det_str == truth_str:
                count += 1
                print(f"✅ 第{i + 1}句正确: {det_str}")
            else:
                print(f"❌ 第{i + 1}句错误: 检测 '{det_str}' != 真实 '{truth_str}'")

            min_len = min(len(det_str), len(truth_str))
            for j in range(min_len):
                if det_str[j] == truth_str[j]:
                    correct_bits_count += 1
                total_bits_count += 1

            len_diff = abs(len(det_str) - len(truth_str))
            total_bits_count += len_diff

            # 计算最终指标
        segment_acc = count / len(detected_segments) if len(detected_segments) > 0 else 0
        bit_acc = correct_bits_count / total_bits_count if total_bits_count > 0 else 0

        print(f"整句准确率 (Segment Acc): {segment_acc:.2%}")
        print(f"位匹配率 (Bit Acc): {bit_acc:.2%}")

        return {"is_watermarked": True, "score": bit_acc, "score_match": segment_acc}

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization for Pmark."""
        from nltk.tokenize import sent_tokenize

        # 1. 句子切分与模型初始化
        sentences = sent_tokenize(text)

        L = self.config.dimension_d
        lsh_model = self.utils.SBERTLSHModel(lsh_model_path=self.config.path_to_embedder, batch_size=1, lsh_dim=L,
                                             sbert_type='base')
        normals = torch.tensor(lsh_model.hasher.normals, device=self.config.device).float()
        target_normals = normals[:L]

        sent_highlights = []
        prompt = kwargs.get("prompt", "")

        # 2. 逐句检测水印匹配情况 (复用 detect_watermark 逻辑)
        for i, target_sentence in enumerate(sentences):
            if i == 0:
                context_window = [prompt] if prompt else [""]
            else:
                start_idx = max(0, i - 1)
                context_window = sentences[start_idx:i]

            anchor = self.anchor_generator.get_anchor(context_window, self.config.path_to_centroids)
            current_target_normals = self.orthogonalize_normals(target_normals, anchor)

            target_numpy = lsh_model.get_embeddings([target_sentence])
            target_embed = torch.tensor(target_numpy, device=self.config.device).float()

            scores = torch.matmul(target_embed.float(), current_target_normals.T.float())
            bits = (scores > 0).int().tolist()[0]

            # 与生成时记录的 segments_d 进行对比验证
            expected_bits = self.segments_d[i] if hasattr(self, 'segments_d') and i < len(self.segments_d) else None

            is_match = 0
            if expected_bits is not None:
                truth_str = "".join(map(str, expected_bits)) if isinstance(expected_bits, list) else str(expected_bits)
                det_str = "".join(map(str, bits))
                # 只有当前句提取的 bits 与目标完全一致，该句才高亮为 1 (绿色)
                if det_str == truth_str:
                    is_match = 1

            sent_highlights.append(is_match)

        # 3. 对齐 Token 与句子的高亮状态
        encoded_text = \
            self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
                self.config.device)
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        # 构建字符索引到句子索引的映射，以便准确地将 Token 归属到对应句子
        char_to_sent = {}
        curr_idx = 0
        for i, s in enumerate(sentences):
            start = text.find(s, curr_idx)
            if start != -1:
                for j in range(start, start + len(s)):
                    char_to_sent[j] = i
                curr_idx = start + len(s)

        highlight_values = []
        curr_text_idx = 0
        for token in decoded_tokens:
            token_len = len(token)
            # 取 Token 中点的字符索引来判断归属句子，防止边界空格导致误判
            mid_idx = curr_text_idx + max(0, token_len // 2)
            s_idx = char_to_sent.get(mid_idx, -1)

            if s_idx == -1:
                # 若因多余空格等未匹配到，平滑回退到起始字符映射
                s_idx = char_to_sent.get(curr_text_idx, 0)

            # 写入状态：1 (匹配成功，渲染为绿)，0 (匹配失败/未加水印文本，渲染为红)
            highlight_values.append(sent_highlights[s_idx] if s_idx < len(sent_highlights) else 0)
            curr_text_idx += token_len

        # Pmark 算法并未如 KGW 一样直接干预 Token 的 logits/weight，因此第三个参数 weights 保留默认值 None
        return DataForVisualization(decoded_tokens, highlight_values)

    def orthogonalize_normals(self, normals, anchor):
        """
        让投影矩阵 (normals) 与当前语义中心 (anchor) 正交。
        也就是：在检测水印时，忽略掉图像/文本本身原本就有的方向。

        normals: (L, dim)
        anchor: (1, dim)
        """
        anchor_norm = torch.nn.functional.normalize(anchor, p=2, dim=1)  # (1, dim)

        projections = torch.matmul(normals.float().cuda(), anchor_norm.T.float().cuda())

        # 3. 从 normals 中减去这个分量 (Reject)，排除掉anchor_norm方向的影响
        normals_ortho = normals.cuda() - projections * anchor_norm.cuda()

        # 4. 重新归一化 normals (可选，但推荐以保持 scale 一致)
        normals_ortho = torch.nn.functional.normalize(normals_ortho, p=2, dim=1)

        return normals_ortho


class RobustAnchorGenerator:
    """
    负责生成抗干扰的 Anchor 向量。
    支持四种模式：
    1. 'baseline': 原始逻辑 (直接使用上下文 Embedding，脆弱)
    2. 'keyword': 关键词排序模式 (抗语序变化、抗修饰词增删)
    3. 'cluster': 语义聚类/量化模式 (抗微小语义漂移，基于 K-Means)
    4. 'topk_sign': 显著语义二值化 (SSB) 模式 (无需训练，平滑降级，理论最强)
    """

    def __init__(self, mode='topk_sign', embedder=None, device='cuda', top_k=64):
        self.mode = mode
        self.embedder = embedder
        self.device = device
        self.top_k = top_k  # SSB 模式下的 Top-K 维度数量
        self.cluster_centers = None
        self.k = 3
        self.temperature = 1

    def get_anchor(self, text_list: List[str], cluster_centers) -> torch.Tensor:
        """
        输入: 上下文句子列表
        输出: (1, dim) 的 Anchor Tensor
        """
        if not text_list:
            dim = self.embedder.dimension if hasattr(self.embedder, 'dimension') else 768
            return torch.zeros((1, dim), device=self.device)

        if self.mode == 'topk_sign':
            # === 新方法: Top-K Salient Semantic Binarization (SSB) ===
            last_sentence = text_list[-1]

            # 1. 获取原始连续向量
            raw_emb = self.embedder.get_embeddings([last_sentence])[0]
            raw_emb_tensor = torch.tensor(raw_emb, dtype=torch.float, device=self.device).unsqueeze(0)  # (1, dim)

            # 动态调整 top_k 以防 embedder 维度较小
            current_k = min(self.top_k, raw_emb_tensor.size(1))

            # 2. 寻找最显著的语义维度（绝对值最大的 Top-K）
            magnitudes = torch.abs(raw_emb_tensor)
            _, topk_indices = torch.topk(magnitudes, current_k, dim=1)

            # 3. 创建高度鲁棒的离散稀疏锚点（仅保留 Top-K 维度的符号）
            anchor = torch.zeros_like(raw_emb_tensor)
            signs = torch.sign(raw_emb_tensor.gather(1, topk_indices))

            # 边缘情况：如果正好为 0，默认给个正号保证稳定性
            signs[signs == 0] = 1.0

            # 将离散符号映射回原来的维度位置
            anchor.scatter_(1, topk_indices, signs)

            # 4. L2 归一化，使其完美适配后续的正交子空间投影
            weighted_anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)

            return weighted_anchor

        elif self.mode == 'keyword':
            last_sentence = text_list[-1]
            keywords = self._extract_keywords(last_sentence)
            keywords = sorted(keywords)
            pseudo_text = " ".join(keywords)
            emb = self.embedder.get_embeddings([pseudo_text])[0]
            return torch.tensor(emb, device=self.device).unsqueeze(0)

        elif self.mode == 'cluster':
            # 性能优化：只有在 cluster 模式且还没加载时才去读文件
            if self.cluster_centers is None and cluster_centers is not None:
                self.cluster_centers = torch.load(cluster_centers)

            last_sentence = text_list[-1]
            raw_emb = self.embedder.get_embeddings([last_sentence])[0]
            raw_emb_tensor = torch.tensor(raw_emb, device=self.device).unsqueeze(0)  # (1, dim)

            # 计算距离: ||x - c||^2
            dists = torch.cdist(raw_emb_tensor.to(self.device),
                                self.cluster_centers.to(self.device))  # (1, num_clusters)
            closest_idx = torch.argmin(dists, dim=1).item()

            weighted_anchor = self.cluster_centers[closest_idx].unsqueeze(0)

            return weighted_anchor

        else:
            full_text = " ".join(text_list)
            emb = self.embedder.get_embeddings([full_text])[0]
            return torch.tensor(emb, device=self.device).unsqueeze(0)

    def _extract_keywords(self, text):
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)

        keywords = [word for word, tag in tags
                    if (tag.startswith('NN') or tag.startswith('VB')) and len(word) > 1]

        if not keywords:
            return tokens
        return keywords
