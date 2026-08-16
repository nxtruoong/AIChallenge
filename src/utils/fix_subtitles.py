import json
from pathlib import Path
import re

def fix_subtitles(subtitle_dir):
    dir_path = Path(subtitle_dir)
    if not dir_path.exists():
        print(f"Directory {subtitle_dir} not found.")
        return

    # A list of tuples (pattern, replacement)
    # Using regex for case-insensitive replacements where necessary
    replacements = [
        (re.compile(r'Asinomoto|Aginomoto|Agino Moto|Ajino Moto', re.IGNORECASE), 'Ajinomoto'),
        (re.compile(r'mã queer|mã quét', re.IGNORECASE), 'mã QR'),
        (re.compile(r'asin ngon|ajin ngon|agin ngon|ajinonnam', re.IGNORECASE), 'Aji-ngon'),
        (re.compile(r'náo chuẩn', re.IGNORECASE), 'nấu chuẩn'),
        (re.compile(r'đề trợ', re.IGNORECASE), 'tài trợ'),
        (re.compile(r'soi', re.IGNORECASE), 'xôi'),
        (re.compile(r'mông cải', re.IGNORECASE), 'bông cải'),
        (re.compile(r'hạm nhân', re.IGNORECASE), 'hạnh nhân'),
        (re.compile(r'vầy mặt luôn', re.IGNORECASE), 'giùm anh luôn'),
        (re.compile(r'tầm bớt', re.IGNORECASE), 'tầm bóp')
    ]

    fixed_count = 0
    file_count = 0

    for json_file in dir_path.glob("*.json"):
        file_count += 1
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        for segment in data:
            original_text = segment.get("text", "")
            new_text = original_text
            for pattern, repl in replacements:
                # regex replace
                if pattern.search(new_text):
                    new_text = pattern.sub(repl, new_text)
                    
            if original_text != new_text:
                segment["text"] = new_text
                changed = True

        if changed:
            fixed_count += 1
            # Atomic write
            tmp_path = json_file.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(json_file)

    print(f"Scanned {file_count} files, fixed issues in {fixed_count} files.")

if __name__ == "__main__":
    fix_subtitles("DAKE_output/extracted_subtitles")
