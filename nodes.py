import os
import traceback

import librosa
import soundfile
from tqdm import tqdm
import folder_paths
import torch
from funasr import AutoModel
from faster_whisper import WhisperModel
path_asr  = 'custom_nodes/ASR/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
path_vad  = 'custom_nodes/ASR/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc = 'custom_nodes/ASR/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
path_asr  = path_asr  if os.path.exists(path_asr)  else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad  = path_vad  if os.path.exists(path_vad)  else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

model = AutoModel(
    model               = path_asr,
    model_revision      = "v2.0.4",
    vad_model           = path_vad,
    vad_model_revision  = "v2.0.4",
    punc_model          = path_punc,
    punc_model_revision = "v2.0.4",
)

class ASRNode:
    """

    属性
    ----------
    RETURN_TYPES (`tuple`):
        Specifies the types of the outputs of the node.
    RETURN_NAMES (`tuple`):
        Specifies the names of the outputs of the node.
    FUNCTION (`str`):
        Specifies the name of the entry point method for the node.
    OUTPUT_NODE ([`bool`]):
        Specifies if this node is an output node.
    CATEGORY (`str`):
        Specifies the category to which this node belongs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        This method returns a dictionary specifying the input types and configurations.

        Returns:
            dict: A dictionary containing the input types and their respective configurations.
        """
        ft_language_list = ["zh", "en", "ja", "auto"]
        return {
            "required": {
                "input_folder": ("AUDIO",),
                "model_size": (["large-v3", "large-v1", "medium.en", "small.en", "tiny.en"], {
                    "default": "large-v3"
                }),
                "language": (ft_language_list, {
                    "default": "auto"
                }),
                "precision": (["float16", "float32"], {
                    "default": "float16"
                }),
                "carry_over_folder": (["true", "false"], {
                         "default": "true"
                     }),
            },
        }

    RETURN_TYPES = ("AUDIO", "ASR", "LANGUAGE")
    RETURN_NAMES = ("audio_path", "asr_path","language")
    FUNCTION = "execute_asr"
    CATEGORY = "AudioProcessing"

    def __init__(self):
        pass

    def execute_asr(self, input_folder, model_size, language, precision, carry_over_folder):
        input_file_names = os.listdir(input_folder)
        input_file_names.sort()

        self_output_path = output_path

        if carry_over_folder == "true":
            self_output_path = os.path.dirname(input_folder)
        self_output_path = os.path.join(self_output_path, "asr_opt")
        os.makedirs(self_output_path, exist_ok=True)

        # model_path = f'D:\code\ComfyUI\custom_nodes\ASR\models/faster-whisper-{model_size}'
        model_path = f'custom_nodes/ASR/models/faster-whisper-{model_size}'
        # if os.path.exists(f'tools/asr/models/faster-whisper-{model_size}'):

        # if '-local' in model_size:
        #     model_size = model_size[:-6]
        #     model_path = f'models/faster-whisper-{model_size}'
        # else:
        #     model_path = model_size
        if language == 'auto':
            language = None  # 不设置语种由模型自动输出概率最高的语种
        print("loading faster whisper model:", model_size, model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = WhisperModel(model_path, device=device, compute_type=precision)
        except:
            return print(traceback.format_exc())

        output = []
        output_file_name = os.path.basename(input_folder)

        for file_name in tqdm(input_file_names):
            try:
                file_path = os.path.join(input_folder, file_name)
                segments, info = model.transcribe(
                    audio=file_path,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700),
                    language=language)
                text = ''

                if info.language == "zh":
                    print("检测为中文文本, 转 FunASR 处理")
                    text = only_asr(file_path)

                if text == '':
                    for segment in segments:
                        text += segment.text

                output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{text}")

            except:
                return print(traceback.format_exc())

        self_output_path = self_output_path or "output/asr_opt"
        os.makedirs(self_output_path, exist_ok=True)
        output_file_path = os.path.abspath(f'{self_output_path}/{output_file_name}.list')

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output))
            print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
        return  (input_folder,output_file_path,language)


# Helper function for RMS calculation

# Required external utilities
def hf_hub_download(*args, **kwargs):
    pass


def device():
    return 'cpu'


def is_half():
    return False

def only_asr(input_file):
    try:
        # 检查采样率
        sample_rate = check_sample_rate(input_file)

        # 如果不是16k，进行重采样
        if sample_rate != 16000:
            ensure_16k_sample_rate(input_file)

        # text = model.generate(input=input_file)[0]["text"]
        res = model.generate(input=input_file,
                             batch_size_s=300)
        print(res)
    except:
        res = ''
        print(traceback.format_exc())
    return res


def check_sample_rate(file_path):
    with soundfile.SoundFile(file_path) as f:
        sr = f.samplerate
        print(f"The sample rate of the audio file is: {sr} Hz")
    return sr

def ensure_16k_sample_rate(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    soundfile.write(file_path, audio, 16000)
    print(f"Audio file resampled to 16k and saved at: {file_path}")




# Example of userset values; output_path and weights_path must be defined.
output_path =folder_paths.get_output_directory()


# This function is obtained from librosa.





