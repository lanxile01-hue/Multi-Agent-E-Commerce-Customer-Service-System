# 并夕夕 产品SKU编码规则

## 一、SKU编码结构

并夕夕平台所有商品采用 18 位统一 SKU 编码，格式为：`XXX-XXXXXX-XXX-XXX-XXXX`

| 段位 | 含义 | 示例 |
|------|------|------|
| 前3位 | 一级类目代码 | ELC=电子, CLT=服装, FOD=食品 |
| 中间6位 | 二级类目+品牌 | ELC-APL001=苹果电子 |
| 第3段 | 规格/型号代码 | WHI=白色, BLK=黑色, 128=128GB |
| 第4段 | 仓库/产地代码 | SHG=上海仓, SZX=深圳仓 |
| 末4位 | 顺序编号 | 0001-9999 |

## 二、类目代码对照表

### 2.1 电子数码 (ELC)
- ELC-APL001: Apple 品牌
- ELC-HUA002: 华为品牌
- ELC-XIA003: 小米品牌
- ELC-SAM004: 三星品牌
- ELC-SON005: 索尼品牌

### 2.2 服装鞋帽 (CLT)
- CLT-NIK001: Nike 品牌
- CLT-ADI002: Adidas 品牌
- CLT-UNI003: 优衣库品牌
- CLT-ZAR004: ZARA 品牌
- CLT-LIN005: 李宁品牌

### 2.3 食品生鲜 (FOD)
- FOD-SAM001: 三只松鼠品牌
- FOD-WAN002: 旺旺品牌
- FOD-MEN003: 蒙牛品牌
- FOD-YIL004: 伊利品牌
- FOD-FRE005: 鲜丰水果

### 2.4 家居家装 (HOM)
- HOM-IKE001: 宜家品牌
- HOM-MUJ002: 无印良品
- HOM-SUP003: 苏泊尔品牌
- HOM-HAI004: 海尔品牌
- HOM-MID005: 美的品牌

## 三、商品规格特殊编码

| 编码 | 含义 |
|------|------|
| F-SKU | 预售商品标识 |
| L-SKU | 限时秒杀商品 |
| V-SKU | VIP专享商品 |
| B-SKU | 大件物流商品 |
| R-SKU | 跨境电商商品 |

## 四、库存状态代码

- STK-AVAILABLE: 现货充足（库存>100）
- STK-LOW: 库存紧张（库存 1-100）
- STK-OUT: 缺货待补
- STK-DISC: 已停产清仓
- STK-PRE: 预售未到货