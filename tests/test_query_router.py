from app.services.query_router import should_use_manual_rag


def test_routes_generic_customer_service_questions_without_manual_rag():
    questions = (
        "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？",
        "我想退货，但是已经超过7天无理由退换货期限了，还能退吗？",
        "快递丢失了，怎么办？这种情况你们会怎么赔偿？",
        "请问你们的商品能提供纸质版说明书吗？电子版在哪里可以找到？",
        "我购买的商品使用一次就坏了，怎么申请售后维修？",
        "我购买的大家电需要上门安装，但是安装人员要求额外收取费用，该怎么处理？",
        "我购买的大型设备需要上门检修，检修人员说需要拉回仓库维修，该怎么处理？",
    )

    assert all(not should_use_manual_rag(question) for question in questions)


def test_routes_explicit_products_and_manual_operations_to_rag():
    questions = (
        "如何给蓝牙激光鼠标安装电池？",
        "功能键盘的退款政策是什么？",
        "DCB112 指示灯闪烁是什么意思？",
        "根据手册，急转弯时应该如何操作？",
        "如何安装电池？",
        "Manual27_1 展示了什么？",
        "How do I use the battery conversion feature before sailing?",
        "What is the max load of the jetski?",
    )

    assert all(should_use_manual_rag(question) for question in questions)


def test_product_evidence_overrides_ambiguous_customer_terms():
    assert should_use_manual_rag("功能键盘可以退货吗？")
    assert should_use_manual_rag("蓝牙激光鼠标怎么维修？")


def test_uploaded_image_routes_to_manual_rag():
    assert should_use_manual_rag("这张图是什么意思？", has_images=True)
