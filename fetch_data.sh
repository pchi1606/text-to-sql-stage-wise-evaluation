#!/usr/bin/env bash
# Fetch the public benchmark datasets used by this reproduction package.
# Skips any dataset already present. Nothing here is bundled, per the
# datasets' own licenses (Spider: CC BY-SA 4.0; BIRD Mini-Dev: CC BY-NC 4.0
# — academic, non-commercial use).
#
#   Spider questions : HuggingFace xlangai/spider (parquet, sha256-pinned)
#   Spider databases : official spider.zip via the Yale Spider page
#                      (Google Drive; only concert_singer + car_1 extracted)
#   BIRD Mini-Dev    : official BIRD Mini-Dev package (SQLite databases)
set -euo pipefail
cd "$(dirname "$0")"

echo "== Spider questions (parquet, 126 KB) =="
mkdir -p spider/data
if [ -f spider/data/validation-00000-of-00001.parquet ]; then
  echo "already present, skipping"
else
  curl --speed-limit 50000 --speed-time 60 -L -o spider/data/validation-00000-of-00001.parquet \
    "https://huggingface.co/datasets/xlangai/spider/resolve/main/spider/validation-00000-of-00001.parquet"
fi
echo "verifying sha256..."
echo "c3e2a46303899a2d4afe3f6a3a62e59f8d589f241b3cbfb52356479b1f054888  spider/data/validation-00000-of-00001.parquet" | sha256sum -c -

echo "== Spider databases (concert_singer + car_1 only) =="
if [ -f spider/data/db_extract/spider_data/database/concert_singer/concert_singer.sqlite ] && \
   [ -f spider/data/db_extract/spider_data/database/car_1/car_1.sqlite ]; then
  echo "already present, skipping"
else
  echo "downloading official spider.zip (~200 MB)..."
  mkdir -p ${REPRO_TMPDIR:-/tmp}/spider_dl
  if [ ! -f ${REPRO_TMPDIR:-/tmp}/spider_dl/spider.zip ]; then
  # Google Drive large-file download: parse the virus-scan form for the
  # confirmed download URL (confirm token is session-specific).
  fileid="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
  dl_url=$(python3 - "$fileid" << 'PYEOF'
import re, subprocess, sys
from urllib.parse import urlencode
fileid = sys.argv[1]
h = subprocess.run(
    ["curl", "-s", "-L", f"https://drive.google.com/uc?export=download&id={fileid}"],
    capture_output=True, text=True).stdout
m = re.search(r'<form id="download-form" action="([^"]+)"[^>]*>(.*?)</form>', h, re.S)
if not m:
    print(f"https://drive.google.com/uc?export=download&id={fileid}")
else:
    params = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', m.group(2)))
    print(m.group(1) + "?" + urlencode(params))
PYEOF
)
  curl --speed-limit 50000 --speed-time 60 -L -o ${REPRO_TMPDIR:-/tmp}/spider_dl/spider.zip "$dl_url"
  fi
  echo "verifying sha256..."
  echo "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b  ${REPRO_TMPDIR:-/tmp}/spider_dl/spider.zip" | sha256sum -c -
  mkdir -p spider/data/db_extract
  unzip -q -o ${REPRO_TMPDIR:-/tmp}/spider_dl/spider.zip \
    "spider_data/database/concert_singer/*" "spider_data/database/car_1/*" \
    -d spider/data/db_extract -x "__MACOSX/*"
  rm -rf ${REPRO_TMPDIR:-/tmp}/spider_dl
fi

echo "== BIRD Mini-Dev databases =="
if [ -f bird/data/minidev/MINIDEV/dev_databases/california_schools/california_schools.sqlite ]; then
  echo "already present, skipping"
else
  echo "downloading BIRD Mini-Dev package..."
  mkdir -p ${REPRO_TMPDIR:-/tmp}/bird_dl bird/data/minidev
  if [ ! -f ${REPRO_TMPDIR:-/tmp}/bird_dl/minidev.zip ]; then
  curl --speed-limit 50000 --speed-time 60 -L -o ${REPRO_TMPDIR:-/tmp}/bird_dl/minidev.zip \
    "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
  fi
  unzip -q -o ${REPRO_TMPDIR:-/tmp}/bird_dl/minidev.zip -d ${REPRO_TMPDIR:-/tmp}/bird_dl
  # locate the MINIDEV dir containing dev_databases/, whatever the wrapper is
  src=$(find ${REPRO_TMPDIR:-/tmp}/bird_dl -maxdepth 3 -type d -name dev_databases -print -quit | xargs dirname)
  if [ -z "${src}" ]; then echo "ERROR: dev_databases not found in download"; exit 1; fi
  mkdir -p bird/data/minidev/MINIDEV
  cp -r "${src}/dev_databases" bird/data/minidev/MINIDEV/
  if [ -f "${src}/mini_dev_sqlite.json" ]; then
    cp "${src}/mini_dev_sqlite.json" bird/data/minidev/MINIDEV/
  fi
  rm -rf ${REPRO_TMPDIR:-/tmp}/bird_dl
fi

echo "== data ready =="
ls spider/data bird/data/minidev/MINIDEV
