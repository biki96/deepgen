# Data prepration

## Pretrain t2i
the Pretrain t2i config can be found in **`configs/datasets/deepgen_512_fix_pixels/t2i_pretrain.py`**. We use [OpenUni](https://github.com/wusize/OpenUni/blob/main/docs/DATASETS.md)  as our pretrain t2i dataset and following data process steps.

## SFT t2i
The open-source T2I SFT datasets we used are below:
| DATASET        | Download Link                                                |
| ---------- | ------------------------------------------------------------ |
| BLIP3o-60k | https://huggingface.co/datasets/BLIP3o/BLIP3o-60k |
| ShareGPT-4o-Image-T2I | https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image |
| OpenGPT-4o-Image-T2I | https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image |
| Echo-4o-Image | https://huggingface.co/datasets/Yejy53/Echo-4o-Image |
| UniReason-T2I | https://huggingface.co/datasets/Alex11556666/Reason_Tuning |

the banana-50k and text rendering data we will upload 
