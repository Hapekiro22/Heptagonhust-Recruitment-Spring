rem This is the script to run the program on Windows
rem The output will be saved in the results directory

set version=v0.3.1
set output_file=Result\result%version%.txt
set test_file=conf\small.conf
set program=winograd.exe

rem Create the results file
echo. > %output_file%

rem Run the program
.\%program% %test_file% > %output_file%
