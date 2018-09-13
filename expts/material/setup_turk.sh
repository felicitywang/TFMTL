# copy TURK data to the current folder

mkdir -p data/raw
cp -fr /export/a08/fwang/tfmtl/datasets/MATERIAL/TURK/ data/raw

# convert to json
python convert_TURK_to_JSON.py