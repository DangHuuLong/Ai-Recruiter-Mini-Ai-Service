from pathlib import Path


RESUMES_FILE = Path("raw/resumes.jsonl")
# JDS_FILE = Path("raw/job_descriptions.jsonl")
OUTPUT_FILE = Path("current_pair_group.txt")

START_LINE = 706
BATCH_SIZE = 5


def read_jsonl_lines(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_group_to_txt(
    resumes: list[str],
    # job_descriptions: list[str],
    start_line: int,
    output_file: Path,
) -> None:
    end_line = start_line + len(resumes) - 1

    content = []

    content.append(f"GROUP FROM LINE {start_line} TO {end_line}")
    content.append("=" * 80)

    content.append("\nRESUMES")
    content.append("-" * 80)
    for i, resume in enumerate(resumes, start=start_line):
        content.append(f"\n[RESUME LINE {i}]")
        content.append(resume)

    # content.append("\n\nJOB DESCRIPTIONS")
    # content.append("-" * 80)
    # for i, jd in enumerate(job_descriptions, start=start_line):
    #     content.append(f"\n[JD LINE {i}]")
    #     content.append(jd)

    output_file.write_text("\n".join(content), encoding="utf-8")


def main():
    resumes = read_jsonl_lines(RESUMES_FILE)
    # job_descriptions = read_jsonl_lines(JDS_FILE)

    current_line = START_LINE

    while True:
        start_index = current_line - 1
        end_index = start_index + BATCH_SIZE

        resume_batch = resumes[start_index:end_index]
        # jd_batch = job_descriptions[start_index:end_index]

        if not resume_batch:
            print("Đã hết dữ liệu.")
            break

        write_group_to_txt(
            resumes=resume_batch,
            # job_descriptions=jd_batch,
            start_line=current_line,
            output_file=OUTPUT_FILE,
        )

        print(
            f"Đã ghi đè {OUTPUT_FILE} với dữ liệu từ dòng "
            f"{current_line} đến {current_line + len(resume_batch) - 1}"
        )

        current_line += BATCH_SIZE

        user_input = input("Bấm Enter để lấy 5 dòng tiếp theo, hoặc nhập q để thoát: ")

        if user_input.lower().strip() == "q":
            print("Đã dừng.")
            break


if __name__ == "__main__":
    main()