@echo off
cd /d "%~dp0"
title 设备预测性维护系统 - 一键启动

echo ================================================
echo   设备预测性维护系统 - 一键启动
echo ================================================
echo.

echo [1/2] 启动后端（FastAPI，加载模型稍等）...
start "设备维护系统-后端" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo       等待后端就绪...
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ -TimeoutSec 1 | Out-Null; break}catch{Start-Sleep -Seconds 1}}"

echo [2/2] 启动前端（Streamlit）...
start "设备维护系统-前端" cmd /k "python -m streamlit run frontend/app.py --server.port 8501 --server.headless true --theme.primaryColor #EB6127 --theme.backgroundColor #152639 --theme.secondaryBackgroundColor #1C2F44 --theme.textColor #F1DDBC"

echo       等待前端就绪并打开浏览器...
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing http://localhost:8501/ -TimeoutSec 1 | Out-Null; break}catch{Start-Sleep -Seconds 1}}"
start "" http://localhost:8501

echo.
echo 启动完成！
echo   后端接口  http://127.0.0.1:8000/docs
echo   前端看板  http://localhost:8501
echo.
echo 停止时关闭"设备维护系统-后端"和"设备维护系统-前端"两个窗口。
echo.
pause
