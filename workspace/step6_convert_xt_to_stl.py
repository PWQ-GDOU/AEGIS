"""
Step 6: 使用 CAD Assistant 将 .x_t 转换为 STL (GUI 自动化)
"""
import os, sys, time
from pathlib import Path

CAD_EXE = Path(r"workspace/CAD_Assistant\CADAssistant.exe")
DATA_DIR = Path(r"DecPelton2024/DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
OUTPUT_DIR = Path(r"workspace/output")

# 要转换的文件
FILES = [
    (DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t",
     OUTPUT_DIR / "dec_model_runner.stl",
     "模型转轮 D1276P21"),
    (DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t",
     OUTPUT_DIR / "dec_full_runner.stl", 
     "真机转轮 D1247R2"),
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_via_pywinauto():
    """使用 pywinauto 控制 CAD Assistant GUI"""
    from pywinauto.application import Application
    from pywinauto.keyboard import send_keys
    import pywinauto
    
    for xt_path, stl_path, label in FILES:
        if not xt_path.exists():
            print(f"❌ 文件不存在: {xt_path}")
            continue
        
        if stl_path.exists():
            print(f"⏭️ STL 已存在, 跳过: {stl_path.name}")
            continue
        
        print(f"\n🔄 转换: {label}")
        print(f"   输入: {xt_path}")
        print(f"   输出: {stl_path}")
        
        # 启动 CAD Assistant
        app = Application(backend="uia").start(str(CAD_EXE) + f' "{xt_path}"')
        time.sleep(5)  # 等待加载
        
        try:
            # 获取主窗口
            dlg = app.window()
            dlg.wait('visible', timeout=30)
            dlg.set_focus()
            
            # 方法1: Ctrl+E 导出
            send_keys('^e')  # File → Export
            time.sleep(2)
            
            # 在导出对话框中输入文件名
            save_dlg = app.window(title_re=".*Export.*|.*Save.*|.*另存为.*")
            if save_dlg.exists():
                save_dlg.set_focus()
                # 输入路径
                send_keys(str(stl_path))
                time.sleep(1)
                send_keys('{ENTER}')
                time.sleep(3)
            else:
                # 尝试菜单路径: File → Export → STL
                send_keys('%f')  # Alt+F
                time.sleep(0.5)
                send_keys('e')   # Export
                time.sleep(2)
                send_keys(str(stl_path))
                time.sleep(1)
                send_keys('{ENTER}')
                time.sleep(3)
            
            # 关闭 CAD Assistant
            send_keys('%{F4}')
            time.sleep(1)
            
        except Exception as e:
            print(f"   ⚠️ 自动化异常: {e}")
            # 强制关闭
            try:
                app.kill()
            except:
                pass
        
        if stl_path.exists():
            size_mb = stl_path.stat().st_size / 1024 / 1024
            print(f"   ✅ 转换成功! ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ 转换失败!")


def convert_manual_instructions():
    """打印手动转换说明"""
    print("""
╔══════════════════════════════════════════════════════════╗
║  CAD Assistant 手动转换指南                               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. 双击 CADAssistant.exe                                ║
║  2. File → Import → 选择 .x_t 文件                       ║
║  3. File → Export → 格式选 STL                           ║
║  4. 保存到 workspace/output/                             ║
║                                                          ║
║  文件1: 1.1-金窝模型转轮D1276P21_.x_t                     ║
║     → dec_model_runner.stl                               ║
║                                                          ║
║  文件2: 2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t     ║
║     → dec_full_runner.stl                                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    print("="*60)
    print("🔧 批量 .x_t → STL 转换")
    print("="*60)
    
    # 检查pywinauto
    try:
        import pywinauto
        print(f"✅ pywinauto {pywinauto.__version__} 可用")
        auto_available = True
    except ImportError:
        print("⚠️ pywinauto 未安装, 使用手动模式")
        auto_available = False
    
    if auto_available:
        try:
            convert_via_pywinauto()
        except Exception as e:
            print(f"❌ 自动化失败: {e}")
            convert_manual_instructions()
    else:
        convert_manual_instructions()
    
    # 检查结果
    print(f"\n📋 转换结果:")
    for _, stl_path, label in FILES:
        if stl_path.exists():
            size_mb = stl_path.stat().st_size / 1024 / 1024
            print(f"   ✅ {label}: {stl_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {label}: 未生成")


if __name__ == "__main__":
    main()
