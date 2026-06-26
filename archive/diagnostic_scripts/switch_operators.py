"""
算子版本切换工具

用法：
  python scripts/switch_operators.py gray     → 灰度版
  python scripts/switch_operators.py rgb      → RGB 版（三通道取最大）
  python scripts/switch_operators.py colormod → 颜色调制版（可学习1×1颜色卷积）
  python scripts/switch_operators.py list     → 查看当前版本
"""

import sys, shutil
from pathlib import Path

SRC = Path("src")
VERSIONS = {
    "gray":       ("operators_gray.py",       "灰度版"),
    "rgb":        ("operators_rgb.py",        "RGB版"),
    "colormod":   ("operators_colormod.py",   "颜色调制版"),
    "native":     ("operators_native.py",     "颜色原生版"),
    "fixedcolor": ("operators_fixedcolor.py", "固定颜色版"),
}

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/switch_operators.py [gray|rgb|colormod|native|fixedcolor|list]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd in VERSIONS:
        src_name, label = VERSIONS[cmd]
        shutil.copy2(SRC / src_name, SRC / "operators.py")
        print(f"✅ 已切换到 {label} ({src_name} → operators.py)")
    
    elif cmd == "list":
        import filecmp
        if not (SRC / "operators.py").exists():
            print("⚠️  operators.py 不存在")
            return
        for key, (src_name, label) in VERSIONS.items():
            if filecmp.cmp(SRC / "operators.py", SRC / src_name):
                print(f"📄 当前版本: {label} ({src_name})")
                return
        print("⚠️  operators.py 与所有备份都不一致")
    
    else:
        print(f"未知命令: {cmd}")
        print("用法: python scripts/switch_operators.py [gray|rgb|colormod|list]")

if __name__ == "__main__":
    main()
