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

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import re
import copy
import nltk
import torch
import numpy as np
from tqdm import tqdm
from nltk import pos_tag
from nltk.corpus import wordnet
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import DiversityValueError
from evaluation.tools.oracle import QualityOracle
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM
from collections import defaultdict
import json
from itertools import product
import os
import unicodedata
import random
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Actions if char not in alphabet
STRATEGY_LOAD = 1  # load category for this char
STRATEGY_IGNORE = 2  # add char to result
STRATEGY_REMOVE = 3  # remove char from result

ASCII_RANGE = range(128)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCATION = os.path.join(CURRENT_DIR, "homoglyph_data")


class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text


class RandomWalkAttack(TextEditor):
    """
        Remove the watermark using the random walk attack (https://arxiv.org/abs/2311.04378) via black-box access to a quality oracle and a perturbaiton oracle.
        (1) Quality oracle can evaluate whether a candidate output is a high-quality response to a prompt.
        (2) Perturbation oracle can modify an output with a nontrivial probability of maintaining quality, 
            and which induces an efficiently mixing random walk on high-quality outputs.
        
        Examplar Usage: 
        '''
        model_name_or_path="meta-llama/Meta-Llama-3-70B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto') 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        perturbation_oracle = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xl", device_map='auto')
        perturbation_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
        quality_oracle = QualityOracle(tokenizer, model, choice_granularity=5, device=device, check_quality='checker')
        span_length = 6
        attack = RandomWalkAttack(perturbation_tokenizer=perturbation_tokenizer, perturbation_oracle=perturbation_oracle,
                                  quality_oracle=quality_oracle,
                                  max_new_tokens=int(2*span_length), min_length=int(1.5*span_length), 
                                  do_sample=True, top_p=0.95, top_k=None, repetition_penalty=1.5)
        '''
    """

    def __init__(self, perturbation_tokenizer: T5Tokenizer, perturbation_oracle: T5ForConditionalGeneration,
                 quality_oracle: QualityOracle,
                 device='cuda', total_steps=200, span_len=6, target_valid_steps=100, **kwargs):
        """
            Parameters:
            perturbation_tokenizer (T5Tokenizer): The tokenizer for the perturbation oracle.
            perturbation_oracle (T5ForConditionalGeneration): The perturbation oracle.
            quality_oracle (QualityOracle): The quality oracle.
            device (str): The device to use for inference.
            span_len (int): The length of the span to mask in each random walk step.
            total_steps (int): The total number of random walk steps.
            target_valid_steps (int): The target number of valid steps.
        """
        self.perturbation_tokenizer = perturbation_tokenizer
        self.perturbation_oracle = perturbation_oracle.eval()
        self.quality_oracle = quality_oracle
        self.device = device
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        self.span_len = span_len
        self.total_steps = total_steps
        self.target_valid_steps = target_valid_steps
        if self.quality_oracle.check_quality == 'checker':
            from gramformer import Gramformer
            self.gf = Gramformer(models=1, use_gpu=True)

    def perturb(self, text: str):
        final_input_text = self.mask_text(text)

        # Tokenize the input
        final_input = self.perturbation_tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        # Generate the edited text
        with torch.inference_mode():
            outputs = self.perturbation_oracle.generate(**final_input, **self.gen_kwargs)
        outputs = self.perturbation_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        infilled_text = outputs[0]
        final_output_text = final_input_text.replace('<extra_id_0>', infilled_text)

        return final_output_text

    def edit(self, text: str, prompt: str, backtrack_patience: int = 100, max_attempts: int = 1000):
        """Edit the text using the T5 model."""

        original_response, n_response = text, text
        n_iter, valid_steps = 0, 0
        patience = 0
        cached_response = copy.deepcopy(n_response)
        # Process the input text in sentence windows
        pbar = tqdm(total=None)
        while n_iter < self.total_steps or valid_steps < self.target_valid_steps:
            candidate_response = self.perturb(n_response)

            candidate_response = self.grammatical_error_correction(candidate_response)
            candidate_response = self.remove_incomplete_sentences(candidate_response)

            if self.quality_oracle.maintain_quality(prompt, original_response, candidate_response):
                cached_response = n_response
                n_response = candidate_response
                valid_steps += 1
                if valid_steps % 10 == 0:
                    print(f"Original response: {original_response}")
                print(f"Get a better {valid_steps}-th response at step {n_iter}/{self.total_steps}: {n_response}")
                patience = 0
            else:
                patience += 1

            if patience > max_attempts:
                break
            elif patience > backtrack_patience:
                n_response = cached_response
                patience = 0

            pbar.update(1)
            n_iter += 1
        pbar.close()

        return n_response

    def grammatical_error_correction(self, text):
        sentences = sent_tokenize(text)
        corrected_sents = []
        for sent in sentences:
            corrected_sent = self.gf.correct(sent, max_candidates=1).pop()
            corrected_sents.append(corrected_sent)
        corrected_text = ' '.join(corrected_sents)
        return corrected_text

    def mask_text(self, text):
        words = text.replace('\n', ' \n').split(' ')
        if len(words) == 1:
            return text + ' <extra_id_0> '
        start = np.random.randint(0, len(words) - self.span_len)
        end = start + self.span_len
        masked_text = ' '.join(words[:start]) + ' <extra_id_0> ' + ' '.join(words[end:])
        return masked_text

    def contains_verb(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        return any(tag.startswith('VB') for word, tag in tagged_words)

    def remove_incomplete_sentences(self, text):
        sentences = sent_tokenize(text)
        complete_sentences = []
        for sent in sentences:
            if sent.endswith('.') and not self.contains_verb(sent) and not bool(re.match(r'^\d+\.$', sent)):
                continue
            else:
                complete_sentences.append(sent)
        return ' '.join(complete_sentences)

    def correct_text(self, text):
        """Basic punctuation correction"""
        # Replace multiple spaces with a single space
        corrected_text = re.sub(r'\s+', ' ', text)

        # Correct spaces before commas, periods, colons, semicolons, exclamation marks, and question marks
        corrected_text = re.sub(r'\s+([,.;!?])', r'\1', corrected_text)  # Remove space before punctuation
        corrected_text = re.sub(r'([,.;!?])(?!\s)', r'\1 ', corrected_text)  # Ensure space after punctuation if missing

        # Replace multiple occurrences of punctuation marks with a single instance
        # This part targets specific punctuation marks (you can add more as needed)
        corrected_text = re.sub(r'(\.){2,}', '.', corrected_text)
        corrected_text = re.sub(r'(,){2,}', ',', corrected_text)
        corrected_text = re.sub(r'(!){2,}', '!', corrected_text)
        corrected_text = re.sub(r'(\?){2,}', '?', corrected_text)
        corrected_text = re.sub(r'(:){2,}', ':', corrected_text)
        corrected_text = re.sub(r'(;){2,}', ';', corrected_text)

        return corrected_text


class DipperParaphraser(TextEditor):
    """Paraphrase a text using the DIPPER model."""

    def __init__(self, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device='cuda',
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1, **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (T5Tokenizer): The tokenizer for the DIPPER model.
                model (T5ForConditionalGeneration): The DIPPER model.
                device (str): The device to use for inference.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                sent_interval (int): The number of sentences to process at a time.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        # Validate diversity settings
        self._validate_diversity(self.lex_diversity, "Lexical")
        self._validate_diversity(self.order_diversity, "Order")

    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    def edit(self, text: str, reference: str):
        """Edit the text using the DIPPER model."""

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)

        # Preprocess the input text
        text = " ".join(text.split())
        sentences = sent_tokenize(text)

        # Preprocess the reference text
        prefix = " ".join(reference.replace("\n", " ").split())

        output_text = ""

        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])

            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            # Tokenize the input
            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Update the prefix and output text
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


class PegasusParaphraser(TextEditor):
    """
    使用 tuner007/pegasus_paraphrase 进行常规的逐句改写。
    代码保持简洁，仅保留基础的短句过滤，防止模型崩溃。
    """

    def __init__(self, tokenizer, model, device='cuda', sent_interval: int = 1, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.eval().to(device)
        self.device = device
        self.sent_interval = sent_interval

        # 常规改写参数
        self.gen_kwargs = {
            "max_length": 60,  # Pegasus 处理单句一般不超过这个长度
            "num_beams": 5,  # 保证改写质量
            "do_sample": False,  # 关闭随机性，保证结果稳定
            "repetition_penalty": 1.2  # 轻微防止重复
        }
        self.gen_kwargs.update(kwargs)

    def _prompt_is_incomplete(self, prompt: str) -> bool:
        if not prompt: return False
        prompt = prompt.strip()
        return len(prompt) > 0 and not prompt.endswith(('.', '!', '?'))

    def edit(self, text: str, reference: str = ""):
        if not text:
            return text

        sentences = sent_tokenize(text)
        if not sentences:
            return text

        # 保持你原有的首句跳过逻辑
        skip = False
        first_char_match = re.search(r'[A-Za-z0-9]', sentences[0])
        if first_char_match and not first_char_match.group().isupper():
            if len(sentences) >= 2 and self._prompt_is_incomplete(reference):
                skip = True

        edited_sentences = []

        # 按窗口逐句处理
        for i in range(0, len(sentences), self.sent_interval):
            window_sentences = sentences[i: i + self.sent_interval]
            input_text = " ".join(window_sentences)

            # 1. 判定是否跳过首句
            if i == 0 and skip:
                edited_sentences.append(input_text)
                continue

            # 2. 基础短句过滤：如果只有 1 个或更少的单词（比如 "1.", "IBM"），不改写
            tokens = re.split(r'(\w+)', input_text)
            valid_word_count = sum(1 for t in tokens if any(c.isalpha() for c in t))
            if valid_word_count <= 1:
                edited_sentences.append(input_text)
                continue

            # 3. 常规模型推理
            inputs = self.tokenizer([input_text], truncation=True, padding='longest', return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **self.gen_kwargs)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            # 4. 简单的结果清洗
            if not decoded:
                edited_sentences.append(input_text)
            else:
                if not re.search(r'[.!?]$', decoded):
                    decoded += "."

                edited_sentences.append(decoded)
        final_text = " ".join(edited_sentences)
        b = sent_tokenize(final_text)
        return final_text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def _prompt_is_incomplete(self, prompt: str) -> bool:
        prompt = prompt.strip()
        return len(prompt) > 0 and not prompt.endswith(('.', '!', '?'))

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""
        skip = False

        if not text:
            return text

        sentences = sent_tokenize(text)

        first_char = re.search(r'[A-Za-z0-9]', sentences[0]).group()

        if first_char.isupper():
            pass
        else:
            if len(sentences) >= 2 and self._prompt_is_incomplete(reference):
                # merged_sentence = sentences[0].rstrip() + " " + sentences[1].lstrip()
                # sentences = [merged_sentence] + sentences[2:]
                # sentences.pop(0)
                skip = True
        edited_sentences = []

        for i, sent in enumerate(sentences):
            if i == 0 and skip:
                edited_sentences.append(sent)
                continue
            word_list = sent.split()
            valid_word_count = sum(1 for t in word_list if any(c.isalpha() for c in t))
            if valid_word_count <= 1:
                edited_sentences.append(sent)
                continue
            if not word_list:
                continue
            last_token = word_list[-1]
            match = re.search(r'^(.*?)(\W+)$', last_token)
            if match:
                last_word_content = match.group(1)  # 例如 "test." -> "test"
                ending_punct = match.group(2)
                if last_word_content:
                    word_list[-1] = last_word_content
                else:
                    word_list.pop()
            else:
                ending_punct = ""

            kept_words = [word for word in word_list if random.random() >= self.ratio]
            if not kept_words:
                new_sent = ending_punct if ending_punct else " "
            else:
                new_sent = " ".join(kept_words) + ending_punct
            edited_sentences.append(new_sent)
        deleted_text = ' '.join(edited_sentences)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)

        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)

            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')

    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i + 1:]
            masked_text = " ".join(masked_sentence)

            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1

        replaced_text = ' '.join(words)

        return replaced_text


class TruncatePromptTextEditor(TextEditor):
    """Truncate the prompt from the text."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the prompt from the text."""
        if reference is not None:
            truncated_text = ' '.join(text.split()[len(reference.split()):])
            # print(len(sent_tokenize(truncated_text)) + 1)
            return truncated_text
        else:
            return text


class TruncateTaskTextEditor(TextEditor):
    """Truncate the task description from the text, used in code generation."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the task description from the text."""
        if reference is not None:
            truncated_text = text[len(reference):]
            return truncated_text
        else:
            return text


class CodeGenerationTextEditor(TextEditor):
    """Process the code generation output, removing the extra parts."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Process the code generation output, removing the extra parts."""
        text = text.lstrip("\n")
        text = text.split("\n\n")[0]
        return text


class BackTranslationTextEditor(TextEditor):
    """Translate text from source language to intermediary language, then back to the source language."""

    def __init__(self,
                 translate_to_intermediary=Translator(from_lang="en", to_lang="zh").translate,
                 translate_to_source=Translator(from_lang="zh", to_lang="en").translate) -> None:
        """
        Initialize the back translation editor.

        Parameters:
            translate_to_intermediary (function): The function to translate text to the intermediary language.
            translate_to_source (function): The function to translate text to the source language.
        """
        super().__init__()
        self.translate_to_source = translate_to_source
        self.translate_to_intermediary = translate_to_intermediary

    def edit(self, text: str, reference=None):
        intermediary_text = self.translate_to_intermediary(text)
        edit_result = self.translate_to_source(intermediary_text)
        return edit_result


class HomoglyphTextEditor(TextEditor):
    def __init__(self, ratio: float, char_ratio: float = 1.0, categories=("LATIN", "COMMON")):
        self.ratio = ratio
        self.char_ratio = char_ratio
        self.hg = Homoglyphs(categories=categories, strategy=STRATEGY_IGNORE)

    def _prompt_is_incomplete(self, prompt: str) -> bool:
        prompt = prompt.strip()
        return len(prompt) > 0 and not prompt.endswith(('.', '!', '?'))

    def _attack_word(self, word: str) -> str:
        """
        辅助函数：对选中的单词进行同形字替换。
        策略：遍历单词中的每个字符，如果有同形字变体则替换。
        """
        attacked_chars = []
        for char in word:
            # 非字母或空格不处理（虽然 split 后一般不会有空格）
            if ord(char) < 128 and not char.isalpha():
                attacked_chars.append(char)
                continue
            if random.random() > self.char_ratio:
                attacked_chars.append(char)
                continue
            variants = self.hg._get_char_variants(char)
            if variants and not (len(variants) == 1 and variants[0] == char):
                candidates = [v for v in variants if v != char]
                if candidates:
                    attacked_chars.append(random.choice(candidates))
                else:
                    attacked_chars.append(char)
            else:
                attacked_chars.append(char)
        return "".join(attacked_chars)

    def edit(self, text: str, reference=None) -> str:
        skip = False
        if not text:
            return text
        sentences = sent_tokenize(text)
        if not sentences:
            return text
        first_char_match = re.search(r'[A-Za-z0-9]', sentences[0])
        if first_char_match:
            first_char = first_char_match.group()
            if not first_char.isupper():
                if len(sentences) >= 2 and self._prompt_is_incomplete(reference):
                    skip = True

        edited_sentences = []

        # 3. 逐句处理
        for i, sent in enumerate(sentences):
            if i == 0 and skip:
                edited_sentences.append(sent)
                continue
            tokens = re.split(r'(\w+)', sent)
            valid_word_count = sum(1 for t in tokens if any(c.isalpha() for c in t))
            if valid_word_count <= 1:
                edited_sentences.append(sent)
                continue
            attacked_tokens = []
            for token in tokens:
                if not token:
                    continue
                is_word = any(c.isalpha() for c in token)
                if is_word and random.random() < self.ratio:
                    attacked_tokens.append(self._attack_word(token))
                else:
                    attacked_tokens.append(token)
            edited_sentences.append("".join(attacked_tokens))
        new_sentence = " ".join(edited_sentences)
        return new_sentence


class Categories:
    fpath = os.path.join(DATA_LOCATION, "categories.json")

    @classmethod
    def _get_ranges(cls, categories):
        """
        :return: iter: (start code, end code)
        :rtype: list
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        for category in categories:
            if category not in data["aliases"]:
                raise ValueError("Invalid category: {}".format(category))

        for point in data["points"]:
            if point[2] in categories:
                yield point[:2]

    @classmethod
    def get_alphabet(cls, categories):
        """
        :return: set of chars in alphabet by categories list
        :rtype: set
        """
        alphabet = set()
        for start, end in cls._get_ranges(categories):
            chars = (chr(code) for code in range(start, end + 1))
            alphabet.update(chars)
        return alphabet

    @classmethod
    def detect(cls, char):
        """
        :return: category
        :rtype: str
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        try:
            category = unicodedata.name(char).split()[0]
        except (TypeError, ValueError):
            # In Python2 unicodedata.name raise error for non-unicode chars
            # Python3 raise ValueError for non-unicode characters
            pass
        else:
            if category in data["aliases"]:
                return category

        code = ord(char)
        for point in data["points"]:
            if point[0] <= code <= point[1]:
                return point[2]

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data["aliases"])


class Languages:
    fpath = os.path.join(DATA_LOCATION, "languages.json")

    @classmethod
    def get_alphabet(cls, languages):
        """
        :return: set of chars in alphabet by languages list
        :rtype: set
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        alphabet = set()
        for lang in languages:
            if lang not in data:
                raise ValueError("Invalid language code: {}".format(lang))
            alphabet.update(data[lang])
        return alphabet

    @classmethod
    def detect(cls, char):
        """
        :return: set of languages which alphabet contains passed char.
        :rtype: set
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        languages = set()
        for lang, alphabet in data.items():
            if char in alphabet:
                languages.add(lang)
        return languages

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.keys())


class Homoglyphs:
    def __init__(
            self,
            categories=None,
            languages=None,
            alphabet=None,
            strategy=STRATEGY_IGNORE,
            ascii_strategy=STRATEGY_IGNORE,
            ascii_range=ASCII_RANGE,
    ):
        # strategies
        if strategy not in (STRATEGY_LOAD, STRATEGY_IGNORE, STRATEGY_REMOVE):
            raise ValueError("Invalid strategy")
        self.strategy = strategy
        self.ascii_strategy = ascii_strategy
        self.ascii_range = ascii_range

        # Homoglyphs must be initialized by any alphabet for correct work
        if not categories and not languages and not alphabet:
            categories = ("LATIN", "COMMON")

        # cats and langs
        self.categories = set(categories or [])
        self.languages = set(languages or [])

        # alphabet
        self.alphabet = set(alphabet or [])
        if self.categories:
            alphabet = Categories.get_alphabet(self.categories)
            self.alphabet.update(alphabet)
        if self.languages:
            alphabet = Languages.get_alphabet(self.languages)
            self.alphabet.update(alphabet)
        self.table = self.get_table(self.alphabet)

    @staticmethod
    def get_table(alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def get_restricted_table(source_alphabet, target_alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in source_alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in target_alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def uniq_and_sort(data):
        result = list(set(data))
        result.sort(key=lambda x: (-len(x), x))
        return result

    def _update_alphabet(self, char):
        # try detect languages
        langs = Languages.detect(char)
        if langs:
            self.languages.update(langs)
            alphabet = Languages.get_alphabet(langs)
            self.alphabet.update(alphabet)
        else:
            # try detect categories
            category = Categories.detect(char)
            if category is None:
                return False
            self.categories.add(category)
            alphabet = Categories.get_alphabet([category])
            self.alphabet.update(alphabet)
        # update table for new alphabet
        self.table = self.get_table(self.alphabet)
        return True

    def _get_char_variants(self, char):
        if char not in self.alphabet:
            if self.strategy == STRATEGY_LOAD:
                if not self._update_alphabet(char):
                    return []
            elif self.strategy == STRATEGY_IGNORE:
                return [char]
            elif self.strategy == STRATEGY_REMOVE:
                return []

        alt_chars = self.table.get(char, set())
        if alt_chars:
            alt_chars2 = [self.table.get(alt_char, set()) for alt_char in alt_chars]
            alt_chars.update(*alt_chars2)
        alt_chars.add(char)

        # uniq, sort and return
        return self.uniq_and_sort(alt_chars)

    def _get_combinations(self, text, ascii=False):
        variations = []
        for char in text:
            alt_chars = self._get_char_variants(char)

            if ascii:
                alt_chars = [char for char in alt_chars if ord(char) in self.ascii_range]
                if not alt_chars and self.ascii_strategy == STRATEGY_IGNORE:
                    return

            if alt_chars:
                variations.append(alt_chars)
        if variations:
            for variant in product(*variations):
                yield "".join(variant)

    def get_combinations(self, text):
        return list(self._get_combinations(text))

    def _to_ascii(self, text):
        for variant in self._get_combinations(text, ascii=True):
            if max(map(ord, variant)) in self.ascii_range:
                yield variant

    def to_ascii(self, text):
        return self.uniq_and_sort(self._to_ascii(text))
