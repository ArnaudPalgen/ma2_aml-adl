if ! ffmpeg -v error -i $1 -f null - 2> /dev/null ; then
    rm $1
    echo "$1 removed"
fi