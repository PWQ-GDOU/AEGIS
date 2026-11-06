"""
Step 6b: pyautogui 自动化 CAD Assistant .x_t → STL 转换
使用真实键鼠模拟 (非 accessibility API)
"""
import os, sys, time, subprocess
from pathlib import Path
import pyautogui
import pyperclip  # 用于剪贴板操作

CAD_EXE = Path(r"workspace/CAD_Assistant\CADAssistant.exe")
DATA_DIR = Path(r"DecPelton2024/DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
OUTPUT_DIR = Path(r"workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    (DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t",
     OUTPUT_DIR / "dec_model_runner.stl",
     "模型转轮 D1276P21"),
    (DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t",
     OUTPUT_DIR / "dec_full_runner.stl",
     "真机转轮 D1247R2"),
]

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.3


def kill_existing():
    os.system('taskkill /f /im CADAssistant.exe 2>nul >nul')
    time.sleep(2)


def wait_and_focus(title_contains="CAD Assistant", timeout=30):
    """等待窗口出现并尝试聚焦"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            # pyautogui way: find window by title
            windows = pyautogui.getWindowsWithTitle(title_contains)
            if windows:
                win = windows[0]
                try:
                    win.activate()
                    win.maximize()
                except:
                    pass
                time.sleep(1)
                return True
        except:
            pass
        time.sleep(1)
    return False


def paste_path(path_str):
    """通过剪贴板粘贴路径"""
    import pyperclip
    pyperclip.copy(path_str)
    time.sleep(0.2)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.3)


def convert_one(xt_path, stl_path, label):
    """转换单个文件"""
    print(f"\n{'='*50}")
    print(f"🔄 {label}")
    
    if stl_path.exists():
        size_mb = stl_path.stat().st_size / 1024 / 1024
        if size_mb > 0.5:
            print(f"   ✅ 已存在 ({size_mb:.1f} MB), 跳过")
            return True
    
    kill_existing()
    
    # 直接启动 CAD Assistant (不带文件参数, 先打开空程序)
    print(f"   启动 CAD Assistant...")
    subprocess.Popen([str(CAD_EXE)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    
    if not wait_and_focus("CAD Assistant", timeout=15):
        print(f"   ❌ 窗口未出现")
        kill_existing()
        return False
    
    print(f"   窗口已聚焦")
    time.sleep(2)
    
    # === 导入文件: File → Import ===
    print(f"   导入文件...")
    pyautogui.hotkey('ctrl', 'o')  # Open/Import
    time.sleep(2)
    
    # 在打开文件对话框中粘贴路径
    paste_path(str(xt_path.resolve()))
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(15)  # 等待大文件加载 (47MB Parasolid)
    
    print(f"   文件已加载, 开始导出...")
    
    # === 导出 STL ===
    # Ctrl+Shift+E 在很多 CAD 软件中是 Export
    pyautogui.hotkey('ctrl', 'shift', 'e')
    time.sleep(2)
    
    # 如果上面的快捷键不工作, 尝试菜单
    pyautogui.hotkey('alt', 'f')  # File menu
    time.sleep(0.5)
    pyautogui.press('e')  # Export
    time.sleep(2)
    
    # 在导出对话框中粘贴STL路径
    paste_path(str(stl_path.resolve()))
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(5)
    
    # 处理可能的确认对话框
    pyautogui.press('enter')
    time.sleep(2)
    
    # 等待导出完成 (STL文件写入)
    start_wait = time.time()
    while time.time() - start_wait < 30:
        if stl_path.exists() and stl_path.stat().st_size > 10000:
            break
        time.sleep(2)
    
    # 关闭
    pyautogui.hotkey('alt', 'f4')
    time.sleep(1)
    kill_existing()
    
    if stl_path.exists():
        size_mb = stl_path.stat().st_size / 1024 / 1024
        if size_mb > 0.5:
            print(f"   ✅ 成功! ({size_mb:.1f} MB)")
            return True
    print(f"   ❌ 失败")
    return False


def main():
    print("="*60)
    print("🔧 pyautogui 自动化 .x_t → STL")
    print("="*60)
    
    # 检查 pyperclip
    try:
        import pyperclip
    except:
        print("安装 pyperclip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyperclip', '--quiet'])
        import pyperclip
    
    for xt_path, stl_path, label in FILES:
        if not xt_path.exists():
            print(f"❌ 文件不存在: {xt_path}")
            continue
        try:
            convert_one(xt_path, stl_path, label)
        except Exception as e:
            print(f"   ❌ 异常: {e}")
            kill_existing()
    
    # 报告
    print(f"\n{'='*50}")
    print(f"📋 转换结果:")
    for _, stl_path, label in FILES:
        if stl_path.exists():
            size_mb = stl_path.stat().st_size / 1024 / 1024
            print(f"   {'✅' if size_mb > 0.5 else '⚠️'} {label}: {stl_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {label}: 未生成")
    
    kill_existing()


if __name__ == "__main__":
    main()
