#!/bin/bash

MUSESCORE_EXE_PATH="C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

filename=$1
shift

# echo "Got optional args: $@"

# opt_args=()
# while [[ $# -gt 0 ]]; do
#     echo "One arg is $1"
#     if [[ "$1" == "-"* ]]; then
#         echo "Got a flag $1"

#     shift

# done

# opt_args=()
# while getopts ":w:" opt; do
#     case ${opt} in
#         w)
#             opt_args+=("-w")
#             echo "Got -w args: ${OPTARG}"
#             ;;

#     esac
# done
echo "Transcribing sheet music from file: $filename"
python_command="python write_sheet_music.py \"$filename\" $@"
echo "Running command: $python_command"
eval "$python_command"
if [ $? -eq 0 ]; then
    echo "Transcribing to xml file complete, saving pdf sheet music..."
    save_pdf_file="mysheetmusic.pdf"
    "$MUSESCORE_EXE_PATH" -o "$save_pdf_file" "mysheetmusic.xml"
    echo "PDF of sheet music saved to $save_pdf_file."
else
    echo "Encountered unhandled error in writing sheet music."
    echo "Please try a different set of input parameters (extending the sampled window range, for example) and try again."
fi
