"""
Faithfulness + 幻觉率评测：基于 Guardrails 模块，测试 Agent 回复的忠实度和幻觉率。

用法: python -m app.scripts.run_faithfulness_eval
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from app.config.settings import settings
from app.agent.guardrails import Guardrails
from app.agent.tools.manager import ToolManager
from app.agent.tools.registry import execute_tool

# 测试用例：问题 + 预期调用的工具
TEST_CASES = [
    {"query": "查询订单 ORD-20240115-001", "tool": "query_order", "args": {"order_id": "ORD-20240115-001"}},
    {"query": "帮我查一下物流 ORD-20240115-001", "tool": "query_logistics", "args": {"order_id": "ORD-20240115-001"}},
    {"query": "有没有红色的运动鞋", "tool": "query_product", "args": {"keyword": "运动鞋"}},
    {"query": "帮我看看有哪些订单", "tool": "list_user_orders", "args": {}},
    {"query": "退换货政策怎么操作", "tool": "search_knowledge", "args": {"query": "退换货政策怎么操作"}},
    {"query": "我要退货 ORD-20240115-001 质量问题", "tool": "apply_refund", "args": {"order_id": "ORD-20240115-001", "reason": "质量问题"}},
    {"query": "钻石会员有什么福利", "tool": "search_knowledge", "args": {"query": "钻石会员有什么福利"}},
    {"query": "七天无理由退货适用范围", "tool": "search_knowledge", "args": {"query": "七天无理由退货适用范围"}},
    {"query": "商品降价了能不能退差价", "tool": "search_knowledge", "args": {"query": "商品降价了能不能退差价"}},
    {"query": "双11有什么满减活动", "tool": "search_knowledge", "args": {"query": "双11有什么满减活动"}},
    {"query": "运费怎么算的", "tool": "search_knowledge", "args": {"query": "运费怎么算"}},
    {"query": "能不能指定发顺丰", "tool": "search_knowledge", "args": {"query": "指定快递"}},
    {"query": "会员日有什么折扣", "tool": "search_knowledge", "args": {"query": "会员日折扣"}},
    {"query": "我的订单 ORD-20240120-002", "tool": "query_order", "args": {"order_id": "ORD-20240120-002"}},
    {"query": "这个订单能否开发票", "tool": "search_knowledge", "args": {"query": "开发票"}},
    {"query": "商品坏了怎么保修", "tool": "search_knowledge", "args": {"query": "保修"}},
    {"query": "怎么联系人工客服", "tool": "search_knowledge", "args": {"query": "人工客服"}},
    {"query": "订单取消后退款什么时候到", "tool": "search_knowledge", "args": {"query": "取消订单退款"}},
    {"query": "能不能修改订单颜色", "tool": "search_knowledge", "args": {"query": "修改订单"}},
    {"query": "买贵了能不能退差价", "tool": "search_knowledge", "args": {"query": "退差价"}},
    {"query": "有没有包邮的商品推荐", "tool": "query_product", "args": {"keyword": "包邮"}},
    {"query": "帮我查一下热销榜", "tool": "query_product", "args": {"keyword": "热销"}},
    {"query": "你们的退换货地址是什么", "tool": "search_knowledge", "args": {"query": "退换货地址"}},
    {"query": "怎么上传退货凭证", "tool": "search_knowledge", "args": {"query": "退货凭证"}},
    {"query": "我的优惠券怎么没了", "tool": "search_knowledge", "args": {"query": "优惠券过期"}},
    {"query": "有没有学生优惠活动", "tool": "search_knowledge", "args": {"query": "学生优惠"}},
    {"query": "帮我查商品 PROD-001 详情", "tool": "query_product", "args": {"keyword": "PROD-001"}},
    {"query": "店铺评分多少算好的", "tool": "search_knowledge", "args": {"query": "店铺评分"}},
    {"query": "怎么提现余额", "tool": "search_knowledge", "args": {"query": "提现"}},
    {"query": "为什么我的订单被关闭了", "tool": "search_knowledge", "args": {"query": "订单关闭"}},
    {"query": "能不能看看评价再买", "tool": "search_knowledge", "args": {"query": "评价"}},
    {"query": "退货运费险怎么理赔", "tool": "search_knowledge", "args": {"query": "运费险"}},
    {"query": "食品有没有保质期要求", "tool": "search_knowledge", "args": {"query": "食品安全保质期"}},
    {"query": "大件家具怎么配送", "tool": "search_knowledge", "args": {"query": "大件配送"}},
    {"query": "当日达覆盖哪些城市", "tool": "search_knowledge", "args": {"query": "当日达"}},
    {"query": "积分是不是会过期", "tool": "search_knowledge", "args": {"query": "积分过期"}},
    {"query": "签到有什么奖励", "tool": "search_knowledge", "args": {"query": "签到奖励"}},
    {"query": "怎么查看我的会员等级", "tool": "search_knowledge", "args": {"query": "会员等级查看"}},
    {"query": "忘记密码怎么找回", "tool": "search_knowledge", "args": {"query": "找回密码"}},
    {"query": "能不能注销账号", "tool": "search_knowledge", "args": {"query": "注销账号"}},
    {"query": "什么时候发货 ORD-20240115-001", "tool": "query_order", "args": {"order_id": "ORD-20240115-001"}},
    {"query": "有没有夏季新款推荐", "tool": "query_product", "args": {"keyword": "夏季新款"}},
    {"query": "促销券能用几次", "tool": "search_knowledge", "args": {"query": "优惠券使用次数"}},
    {"query": "怎么切换到青少年模式", "tool": "search_knowledge", "args": {"query": "青少年模式"}},
    {"query": "拒收了快递钱什么时候退", "tool": "search_knowledge", "args": {"query": "拒收退款"}},
    {"query": "平台支持哪些支付方式", "tool": "search_knowledge", "args": {"query": "支付方式"}},
    {"query": "怎么申请平台介入处理纠纷", "tool": "search_knowledge", "args": {"query": "平台介入"}},
    {"query": "是不是正品保证", "tool": "search_knowledge", "args": {"query": "正品保证"}},
    {"query": "误删了订单怎么找回", "tool": "search_knowledge", "args": {"query": "找回订单"}},

    # ---- 对抗性用例：工具返回可能为空或不完整，模型容易"脑补" ----
    # 不存在/无效 ID 类
    {"query": "ORD-99999 的物流到哪里了", "tool": "query_logistics", "args": {"order_id": "ORD-99999"}},
    {"query": "我的订单退款多少", "tool": "query_order", "args": {"order_id": "ORD-00000"}},
    {"query": "PROD-888 现在有货吗", "tool": "query_product", "args": {"keyword": "PROD-888"}},
    {"query": "帮我查 ORD-88888888 的物流", "tool": "query_logistics", "args": {"order_id": "ORD-88888888"}},
    {"query": "ORD-20240101-999 能退货吗", "tool": "query_order", "args": {"order_id": "ORD-20240101-999"}},
    {"query": "查一下产品 PROD-XYZ-999", "tool": "query_product", "args": {"keyword": "PROD-XYZ-999"}},
    {"query": "ORD-TEST-000 发货了吗", "tool": "query_order", "args": {"order_id": "ORD-TEST-000"}},
    {"query": "我的订单 ORD-DELETED 状态", "tool": "query_order", "args": {"order_id": "ORD-DELETED"}},

    # 知识库无匹配类
    {"query": "退换货需要几天", "tool": "search_knowledge", "args": {"query": "不存在的政策"}},
    {"query": "有没有限时秒杀 iPhone 15", "tool": "query_product", "args": {"keyword": "iPhone 15"}},
    {"query": "怎么申请 VIP 专属客服", "tool": "search_knowledge", "args": {"query": "VIP客服"}},
    {"query": "我的优惠券过期了还能用吗", "tool": "search_knowledge", "args": {"query": "过期优惠券"}},
    {"query": "能不能货到付款", "tool": "search_knowledge", "args": {"query": "货到付款"}},
    {"query": "怎么申请成为商家", "tool": "search_knowledge", "args": {"query": "商家入驻"}},
    {"query": "跨境商品能不能退换", "tool": "search_knowledge", "args": {"query": "跨境退换"}},
    {"query": "有没有以旧换新活动", "tool": "search_knowledge", "args": {"query": "以旧换新"}},
    {"query": "怎么申请白条分期", "tool": "search_knowledge", "args": {"query": "白条分期"}},
    {"query": "能不能用礼品卡支付", "tool": "search_knowledge", "args": {"query": "礼品卡支付"}},
    {"query": "怎么预约上门安装", "tool": "search_knowledge", "args": {"query": "上门安装"}},
    {"query": "有没有团购优惠", "tool": "search_knowledge", "args": {"query": "团购优惠"}},
    {"query": "怎么绑定亲情账号", "tool": "search_knowledge", "args": {"query": "亲情账号"}},
    {"query": "能不能加急配送", "tool": "search_knowledge", "args": {"query": "加急配送"}},
    {"query": "有没有生日特权", "tool": "search_knowledge", "args": {"query": "生日特权"}},
    {"query": "怎么举报虚假宣传", "tool": "search_knowledge", "args": {"query": "举报虚假宣传"}},

    # 需要具体数值但工具无数据的类（容易编造时间/金额/流程）
    {"query": "退款一般多少个工作日到账", "tool": "search_knowledge", "args": {"query": "退款到账时间"}},
    {"query": "快递超时了赔多少钱", "tool": "search_knowledge", "args": {"query": "快递超时赔偿"}},
    {"query": "退货包运费最多报销多少", "tool": "search_knowledge", "args": {"query": "退货运费报销上限"}},
    {"query": "会员升级需要消费多少钱", "tool": "search_knowledge", "args": {"query": "会员升级消费金额"}},
    {"query": "满减活动最低满多少减多少", "tool": "search_knowledge", "args": {"query": "满减门槛"}},
    {"query": "积分兑换比例是多少", "tool": "search_knowledge", "args": {"query": "积分兑换比例"}},
    {"query": "多长时间不发货可以投诉", "tool": "search_knowledge", "args": {"query": "发货超时投诉"}},
    {"query": "差评达到多少会被下架", "tool": "search_knowledge", "args": {"query": "差评下架标准"}},
]


def generate_reply(client, model, query: str, tool_result: str) -> str:
    """模拟 Agent 生成回复。"""
    prompt = f"""你是电商客服助手，请始终用专业、友好的语气回答用户问题。

以下是系统查询到的相关数据（可能不完整或无结果，仅供参考）:

用户问题: {query}
工具返回: {tool_result}

请直接回复用户。即使数据不完整，也请尽力给出有帮助的回答:"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content or ""


def main():
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    guard = Guardrails(client, settings.model_name, enabled=True)

    total = 0
    hallucination_count = 0
    correction_count = 0
    detector_error_count = 0

    print(f"Faithfulness + 幻觉率评测: {len(TEST_CASES)} 条\n" + "=" * 60)

    for case in TEST_CASES:
        total += 1
        tool_result = execute_tool(case["tool"], case["args"])

        tool_results = [{"name": case["tool"], "arguments": case["args"], "result": tool_result}]
        reply = generate_reply(client, settings.model_name, case["query"], tool_result)

        check = guard.detect_hallucination(reply, tool_results)
        if check.error:
            detector_error_count += 1

        status = "PASS" if not check.has_hallucination else "HALLUCINATION"
        if check.has_hallucination:
            hallucination_count += 1

        print(f"  [{status}] {case['query'][:40]}...")
        if check.has_hallucination:
            for h in check.hallucinations[:2]:
                print(f"        幻觉: {h[:60]}...")
            # 尝试自纠错
            corrected = guard.self_correct(reply, case["query"], tool_results, check.hallucinations)
            recheck = guard.detect_hallucination(corrected, tool_results)
            if not recheck.has_hallucination:
                correction_count += 1
                print(f"        自纠错成功")

    faithfulness = (total - hallucination_count) / total
    hallucination_rate = hallucination_count / total
    correction_rate = correction_count / hallucination_count if hallucination_count else 1.0

    print("\n" + "=" * 60)
    print(f"总测试: {total}")
    print(f"忠实(Faithfulness): {total - hallucination_count}/{total} = {faithfulness:.1%}")
    print(f"幻觉率: {hallucination_count}/{total} = {hallucination_rate:.1%}")
    print(f"自纠错成功率: {correction_count}/{hallucination_count} = {correction_rate:.1%}")
    print(f"自纠错后幻觉率: {(hallucination_count - correction_count)}/{total} = {(hallucination_count - correction_count)/total:.1%}")
    if detector_error_count:
        print(
            f"\n⚠️  检测器异常: {detector_error_count}/{total} 条——这些用例走了「异常→默认无幻觉」"
            "的兜底分支，上面的忠实率/幻觉率不可作为指标使用。"
        )
    else:
        print(f"\n检测器异常: 0/{total}（全部用例的幻觉判断均正常执行）")
    print("\n评测完成。")


if __name__ == "__main__":
    main()
