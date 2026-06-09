from app.services.competition_answer_formatter import format_answer_images


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
