#Requires AutoHotkey v2.0
#SingleInstance Force
InstallKeybdHook
#UseHook


; ----------------------------
; 配置区（根据实际情况修改）
; ----------------------------
; Conda环境的Python绝对路径
PYTHON_PATH := "D:\Codes\Python\condaEnvs\zaishua\python.exe"  ; ← 修改为你的CONDA路径
SCRIPT_PATH := "D:\Codes\Python\ZaiShua2\main.py"            ; ← 修改为你的脚本路径
WORKING_DIR := "D:\Codes\Python\ZaiShua2"             ; ← 修改为你的工作路径
; ----------------------------
; 全局变量
; ----------------------------
pid := 0
exitFlag := false

; ----------------------------
; 主监控逻辑
; ----------------------------
StartPython() {
    global pid := 0
    try {
        ; 使用Conda环境的Python解释器
        Run(PYTHON_PATH ' "' SCRIPT_PATH '"',WORKING_DIR,, &pid)
        SetTimer CheckProcess, 1000*60*1  ; 每1分钟检查一次
        ToolTip "监控已启动（PID: " pid "）"
        SetTimer RemoveToolTip, -2000
    } catch Error as e {
        MsgBox "启动失败: " e.Message
    }
}

; ----------------------------
; F12终止逻辑（穿透BlockInput）
; -------------------------~
~F12:: {
    global exitFlag := true
    SetTimer CheckProcess, 0
    
    ; 终止Python进程
    if pid && ProcessExist(pid) {
        ProcessClose pid
        Sleep 100
    }
    
    ; 强制清理残留进程
    DetectHiddenWindows true
    for proc in WinGetList("python.exe") {
        PostMessage 0x12, 0, 0,, "ahk_id " proc  ; WM_QUIT
    }
    
    ; 确保恢复输入
    DllCall("user32\BlockInput", "Int", 0)
    ; ExitApp
}

; ----------------------------
; 进程监控
; ----------------------------
CheckProcess() {
    global exitFlag, pid
    if exitFlag
        return
    
    if !ProcessExist(pid) {
        StartPython()
        ToolTip "检测到脚本停止，已重启"
        SetTimer RemoveToolTip, -2000
    }
}

RemoveToolTip() {
    ToolTip
}

; ----------------------------
; 初始化
; ----------------------------
Persistent
DllCall("user32\BlockInput", "Int", 0)  ; 确保启动时输入正常
; StartPython()
~F10:: {
    StartPython()
}