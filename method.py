from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.dataset import *
from evaluation.tools.text_editor import *
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, \
    DetectionPipelineReturnType
from transformers import BitsAndBytesConfig
import nltk
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
transformers.logging.set_verbosity_error()   # 只显示错误，不显示警告和信息
# nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('punkt')
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
# cache_dir = r"E:\models\opt-1.3b\models--facebook--opt-1.3b\snapshots\3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
cache_dir =  'facebook/opt-1.3b'
# Load dataset
my_dataset = C4Dataset('dataset/c4/processed_c4.json')
# my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
# my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
# 1. 定义量化配置（12GB 显存运行 7B 模型的必备招式）
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Transformers config
transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(cache_dir).to(device),
                                         tokenizer=AutoTokenizer.from_pretrained(cache_dir),
                                         vocab_size=50272,
                                         device=device,
                                         max_new_tokens=400,
                                         min_length=230,
                                         do_sample=True,
                                         no_repeat_ngram_size=4)

pegasus_tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

pegasus_model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase",
                                                                torch_dtype=torch.float16).to(
    device)

pegasus_editor = PegasusParaphraser(tokenizer=pegasus_tokenizer, model=pegasus_model, device=device, sent_interval=1)

# Load watermark algorithm
myWatermark = AutoWatermark.load('pmark',
                                 algorithm_config='config/Pmark.json',
                                 transformers_config=transformers_config)

torch.cuda.empty_cache()

# Init pipelines
pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset,
                                             text_editor_list=[TruncatePromptTextEditor(), WordDeletion(ratio=0)],
                                             show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

pipeline2 = WatermarkedTextDetectionPipeline(dataset=my_dataset,
                                             text_editor_list=[TruncatePromptTextEditor(), WordDeletion(ratio=0.1)],
                                             show_progress=True, return_type=DetectionPipelineReturnType.SCORES)
pipeline3 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(),
                                                                                   HomoglyphTextEditor(ratio=0.1)],
                                             show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

pipeline4 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(),
                                                                                   SynonymSubstitution(ratio=0.1)],
                                             show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

pipeline5 = WatermarkedTextDetectionPipeline(dataset=my_dataset,
                                             text_editor_list=[TruncatePromptTextEditor(), pegasus_editor],
                                             show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

# pegasus_results = pipeline5.evaluate(myWatermark)

# Evaluate
calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
print(calculator.calculate(pipeline1.evaluate(myWatermark), pipeline2.evaluate(myWatermark)))
