#!/usr/bin/env sh
# take the labels/predictions only

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/t6/mt-4.asr-s5/GOV.tsv | cut -f1,2 > 1ADsGOV

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/tt18/GOV.tsv | cut -f1,2 > 1ADtGOV

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/t6/mt-4.asr-s5/LIF.tsv | cut -f1,2 > 1ADsLIF

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/tt18/LIF.tsv | cut -f1,2 > 1ADtLIF

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/t6/mt-4.asr-s5/LAW.tsv | cut -f1,2 > 1ADsLAW

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/tt18/LAW.tsv | cut -f1,2 > 1ADtLAWdif

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/t6/mt-4.asr-s5/BUS.tsv | cut -f1,2 > 1ADsBUS

awk '$2 == "1"' /export/a08/fwang/tfmtl/expts/material_gold/predictions/doc/1A/DEV/tt18/BUS.tsv | cut -f1,2 > 1ADtBUS

cat 1ADsGOV 1ADtGOV > 1ADGOV

cat 1ADsLIF 1ADtLIF > 1ADLIF

cat 1ADsLAW 1ADtLAW > 1ADLAW

cat 1ADsBUS 1ADtBUS > 1ADBUS

cp 1AGOV d-Government-And-Politics.tsv

sed -i 's/1$/0.5/g' d-Government-And-Politics.tsv

# edit to add Government-And-Politics to the first line

cp 1ALIF d-Lifestyle.tsv

sed -i 's/1$/0.5/g' d-Lifestyle.tsv

# edit

cp 1ABUS d-Business-And-Commerce.tsv

sed -i 's/1$/0.5/g' d-Business-And-Commerce.tsv

# edit

cp 1ALAW d-Law-And-Order.tsv

sed -i 's/1$/0.5/g' d-Law-And-Order.tsv

# edit

# do all of the above for “Sports”

tar -cvzf d-domain.tgz d-Government-And-Politics.tsv d-Lifestyle.tsv d-Business-And-Commerce.tsv d-Law-And-Order.tsv d-Sports.tsv

# upload d-domain.tgz

