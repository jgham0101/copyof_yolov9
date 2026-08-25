
from pathlib import Path

p = Path("train_dual.py")

mark = "# WEEK24_FREEZE_BN_PATCH"

text = p.read_text(
    encoding="utf-8"
)

if mark in text:
    print("already patched")
    raise SystemExit(0)


# train_dual.py에서 epoch마다 호출되는 model.train() 직후에
# frozen BatchNorm을 다시 eval mode로 돌린다.
old = "        model.train()\n"

new = """        model.train()
        # WEEK24_FREEZE_BN_PATCH
        # Keep BatchNorm statistics fixed inside frozen layers.
        if os.getenv("YOLO_FREEZE_BN", "0") == "1" and freeze:
            for _name, _module in model.named_modules():
                if isinstance(
                    _module,
                    torch.nn.modules.batchnorm._BatchNorm
                ):
                    if any(
                        _name == _prefix[:-1]
                        or _name.startswith(_prefix)
                        for _prefix in freeze
                    ):
                        _module.eval()
"""


if old not in text:
    raise RuntimeError(
        "model.train() anchor not found in train_dual.py"
    )


text = text.replace(
    old,
    new,
    1,
)

p.write_text(
    text,
    encoding="utf-8",
)

print("patched:", p)
