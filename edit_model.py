import torch

model_path = "output/gdrn/doorlatch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/model_final.pth"
mod = torch.load(model_path)

mod.pop("optimizer", None)
mod.pop("scheduler", None)
mod['epoch'] = 0
mod['iteration'] = 0

torch.save(mod, "output/gdrn/doorlatch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/model_single_reset.pth")
