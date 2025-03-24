@echo off
rem filepath: d:\High Performance Computing\Heptagon Test\2025-Spring Recruitment\recruitment-2025-spring\run_vtune.bat

set directory=Vtune_Results-Data_Group\vtune_results-v0.3.2

rem 创建结果目录
mkdir %directory% 2>nul

rem 
call "D:\Program Files (x86)\Intel\OneAPI\setvars.bat"

rem 使用VTune运行程序并收集热点数据
vtune -collect hotspots -result-dir %directory%\hotspots .\winograd.exe conf\small.conf

rem vtune -collect uarch-exploration -result-dir vtune_results\uarch .\winograd.exe conf\small.conf

rem 
vtune -collect memory-access -result-dir %directory%\memory .\winograd.exe conf\small.conf

rem 
vtune -collect threading -result-dir %directory%\threading .\winograd.exe conf\small.conf

echo "Performance analysis finished. Please check the results in the vtune_results directory."