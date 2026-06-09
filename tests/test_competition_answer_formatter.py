from app.services.competition_answer_formatter import (
    format_answer_images,
    markdown_tables_to_plain_text,
    markdown_to_plain_text,
)


def test_answer_without_images_is_unchanged() -> None:
    answer = "物流通常会在24小时内完成揽收。"

    assert format_answer_images(answer) == answer


def test_208_battery_installation_images_keep_position_and_order() -> None:
    answer = (
        "按下按钮，电池仓盖弹出。\n"
        "![Manual27_1](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_1.jpg)\n"
        "按照正负极标注装入两节AA电池。\n"
        "![Manual27_2](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_2.jpg)\n"
        "装回电池仓盖。\n"
        "![Manual27_3](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_3.jpg)"
    )

    assert format_answer_images(answer) == (
        '"按下按钮，电池仓盖弹出。\\n<PIC>\\n'
        '按照正负极标注装入两节AA电池。\\n<PIC>\\n'
        '装回电池仓盖。\\n<PIC>",'
        '["Manual27_1", "Manual27_2", "Manual27_3"]'
    )


def test_211_pairing_images_keep_position_and_order() -> None:
    answer = (
        "按下鼠标底部的配对按钮。\n"
        "![Manual27_10](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_10.jpg)\n"
        "按下USB蓝牙接收器底部的按钮。\n"
        "![Manual27_11](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_11.jpg)"
    )

    assert format_answer_images(answer) == (
        '"按下鼠标底部的配对按钮。\\n<PIC>\\n'
        '按下USB蓝牙接收器底部的按钮。\\n<PIC>",'
        '["Manual27_10", "Manual27_11"]'
    )


def test_empty_alt_text_uses_picture_id_from_path() -> None:
    answer = "按下按钮。![](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_10.jpg)"

    assert format_answer_images(answer) == '"按下按钮。<PIC>",["Manual27_10"]'


def test_markdown_is_removed_before_images_are_formatted() -> None:
    answer = (
        "## 电池安装步骤\n\n"
        "**所需电池：** 两节 AA 电池\n\n"
        "> 注意：请确认正负极\n\n"
        "---\n\n"
        "![Manual27_1](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_1.jpg)"
    )

    assert format_answer_images(answer) == (
        '"电池安装步骤\\n\\n'
        '所需电池： 两节 AA 电池\\n\\n'
        '注意：请确认正负极\\n\\n'
        '<PIC>",["Manual27_1"]'
    )


def test_plain_text_cleanup_preserves_ordinary_symbols_and_lists() -> None:
    answer = (
        "错误码 #1001\n"
        "型号 AB-123\n"
        "计算 5 * 10\n"
        "文件 *.json\n"
        "- 检查电池\n"
        "1. 打开电池仓"
    )

    assert markdown_to_plain_text(answer) == answer


def test_plain_text_cleanup_converts_links_but_preserves_images() -> None:
    answer = (
        "访问[官方网站](https://example.com)。\n"
        "![Manual27_1](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_1.jpg)"
    )

    assert markdown_to_plain_text(answer) == (
        "访问官方网站。\n"
        "![Manual27_1](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_1.jpg)"
    )


def test_markdown_table_is_expanded_into_labeled_plain_text_rows() -> None:
    answer = (
        "| 购买渠道 | 退货运费 | 换货运费 |\n"
        "|---------|---------|---------|\n"
        "| **官网订单** | 公司承担往返运费 | 公司承担往返运费 |\n"
        "| **授权经销商** | 用户自行承担 | 公司承担单程运费 |"
    )

    assert markdown_to_plain_text(answer) == (
        "购买渠道：官网订单；退货运费：公司承担往返运费；换货运费：公司承担往返运费\n"
        "购买渠道：授权经销商；退货运费：用户自行承担；换货运费：公司承担单程运费"
    )


def test_table_converter_preserves_non_table_pipe_text() -> None:
    answer = "可选值为 A | B，但型号 AB-123 保持不变。"

    assert markdown_tables_to_plain_text(answer) == answer
