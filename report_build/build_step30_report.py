from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
import json, hashlib

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / 'workspace'
OUT = ROOT / 'report_build'
OUT.mkdir(parents=True, exist_ok=True)
DOCX = OUT / '大容量冲击式水轮机精准抗磨防护技术研究技术报告.docx'

score_path = WS / 'output' / 'step30_competition_self_benchmark_v3_two_sided_lomo_20260909.json'
score = json.loads(score_path.read_text(encoding='utf-8'))


def repo_relative(path):
    """Store portable repository-relative references in report indexes."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd'); tcPr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_text(cell, text, bold=False):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text)); r.bold = bold; r.font.size = Pt(9)
    r.font.name = '宋体'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r.font.color.rgb = RGBColor(0, 0, 0)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


TABLE_NO = 0


def add_table(doc, headers, rows, widths=None):
    global TABLE_NO
    TABLE_NO += 1
    caption = doc.add_paragraph(f'表{TABLE_NO} {headers[0]}与相关内容对应关系')
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_after = Pt(4)
    caption.style = 'Caption'
    for run in caption.runs:
        run.font.name = '宋体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.color.rgb = RGBColor(0, 0, 0)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i,h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], h, True)
    for row in rows:
        cells=t.add_row().cells
        for i,val in enumerate(row): set_cell_text(cells[i], val)
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def add_figure(doc, path, caption):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if path.exists():
        try: p.add_run().add_picture(str(path), width=Inches(6.2))
        except Exception: p.add_run('[图件暂不插入，见图件索引]')
    else: p.add_run('[图件待补充]')
    c = doc.add_paragraph(caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    c.style='Caption'
    for run in c.runs:
        run.font.name='宋体'; run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
        run.font.color.rgb = RGBColor(0, 0, 0)


def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_para(doc, text):
    p=doc.add_paragraph(text); p.paragraph_format.first_line_indent=Cm(0.74); p.paragraph_format.line_spacing=1.35; return p


def add_bullet(doc, text):
    p=doc.add_paragraph(text, style='List Bullet'); p.paragraph_format.line_spacing=1.25; return p


doc=Document()
sec=doc.sections[0]; sec.top_margin=Cm(2.5); sec.bottom_margin=Cm(2.2); sec.left_margin=Cm(2.6); sec.right_margin=Cm(2.2)
styles=doc.styles
for name in ['Normal','Body Text']:
    st=styles[name]; st.font.name='宋体'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); st.font.size=Pt(10.5)
for name,size in [('Title',22),('Heading 1',16),('Heading 2',13),('Heading 3',11.5)]:
    st=styles[name]; st.font.name='宋体'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); st.font.size=Pt(size); st.font.bold=True
styles['Caption'].font.name='宋体'; styles['Caption']._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); styles['Caption'].font.size=Pt(9)
styles['Caption'].font.color.rgb = RGBColor(0, 0, 0)

# Header/footer
header=sec.header.paragraphs[0]; header.text='大容量冲击式水轮机关键过流部件精准抗磨防护技术研究'; header.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer=sec.footer.paragraphs[0]; footer.text='参赛团队：__________    学校/单位：__________'; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER

p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Cm(3)
r=p.add_run('大容量冲击式水轮机关键过流部件精准抗磨防护技术研究'); r.bold=True; r.font.name='宋体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); r.font.size=Pt(22)
for txt in ['赛事技术报告','参赛团队：__________','学校/单位：__________','指导教师：__________','提交日期：2026年9月']:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(18); p.add_run(txt).font.size=Pt(12)
doc.add_page_break()

add_heading(doc,'摘要',1)
add_para(doc,'高水头冲击式水轮机长期运行于含沙高速射流环境，喷嘴口环、喷针和水斗等关键过流部件面临撞击压陷、犁削切削、滑移磨损、空蚀耦合和疲劳损伤等多机制协同作用。局部冲蚀具有明显的空间集中性，传统依靠平均磨损率或经验巡检的方法难以支撑差异化防护设计。针对这一问题，本项目构建了面向 Pelton 水轮机的“机理识别—粒子追踪—DEC 面区映射—守恒面级估计—材料幅值组合—风险分区—防护设计”一体化技术路线。')
add_para(doc,'项目以官方物理几何为边界基础，建立粒子碰撞与关键表面之间的拓扑连接关系，将冲蚀贡献从离散命中事件稳定地映射到有限表面邻域，并通过面积归一化保证总去除质量守恒。针对不同冲蚀机理，统一引入 Oka、Finnie、Tabakoff 和 Mansouri 四类模型，以固定权重形成材料与工况敏感的风险响应。进一步采用五 seed 粒子 ensemble、模型留一评价和 seed 留一评价构建参赛者自建 baseline/worn 参照场。')
add_para(doc,f'结果表明，面级风险图在多粒子规模和多 seed 条件下保持稳定，最大质量归一化 JS 散度不超过 0.05；在参赛者自建验证基准下，保守条件性得分为 {score["conservative_score"]:.2f}/100，名义得分为 {score["nominal_score"]:.2f}/100。该分数用于反映方案在自建基准下的空间识别、排序、分布和幅值一致性，不替代官方真实磨损场。项目最终输出高、中、低风险区分区图和与之对应的材料、结构、维护一体化防护方案，可为大容量冲击式水轮机的抗磨设计、检修计划和寿命评估提供技术支撑。')
add_para(doc,'关键词：冲击式水轮机；含沙高速射流；颗粒冲蚀；Pelton 水斗；面级空间映射；风险分区；抗磨防护；参赛者自建验证基准')

add_heading(doc,'一 赛事要求与方案定位',1)
add_para(doc,'本项目围绕赛事题目“大容量冲击式水轮机关键过流部件精准抗磨防护技术研究”展开，面向高水头、高含沙和长期运行条件下的关键部件冲蚀识别与精准防护问题。方案对应赛事要求的三个层次：第一，系统分析不同冲蚀类型的形成机理及主控因素；第二，建立磨蚀类型、流动参数、泥沙颗粒和部件几何之间的关联模型；第三，形成风险分布图、定量评价指标以及能够指导材料、结构、运行和维护的工程应用方案。')
add_para(doc,'项目的核心定位不是给出单一经验磨损率，而是把局部冲蚀从“不可见的运行损伤”转化为“可计算、可分区、可设计”的工程风险场。通过多物理因素融合和面级守恒映射，预测结果能够直接关联到水斗表面、喷针和喷嘴口环等具体部位，进而支持差异化耐磨层、局部圆角优化、可更换耐磨件和重点巡检策略。')
add_table(doc,['评分维度','方案对应内容','报告章节','主要证据'],[
('设计方案完成程度 20分','机理、算法、材料、验证、分区防护和工程应用闭环','三至十四','赛事方案、程序模块、结果 JSON、图件索引'),
('研发思路合理性 20分','粒子追踪与拓扑面映射、多模型组合、留出评价和不确定性传播','四至十一','收敛报告、自建基准报告、敏感性结果'),
('工程可实现性 20分','风险分区对应材料、结构、检修和寿命评估动作','十三至十四','防护策略矩阵、远程计算和结果封存记录'),
('冲蚀机理识别 20分','撞击、犁削、切削、滑移、空蚀耦合和疲劳关联','三至四','机理分类器、文献证据包、参数先验'),
('空间分布预测 20分','DEC 面区映射、守恒面级估计、热点重叠和排序评价','六至十一','粒子收敛报告、风险图、自建基准评分')], [3.1,5.2,2.1,5.0])

add_heading(doc,'二 工程背景与冲蚀问题定义',1)
add_para(doc,'冲击式水轮机通过喷嘴将高水头势能转化为高速射流动能，射流以较大速度和特定入射角作用于水斗表面。实际河流来水中的石英砂等固体颗粒会随水流进入机组，在射流加速、转轮旋转和水斗曲面导流过程中形成复杂的颗粒—壁面相互作用。冲蚀不仅取决于单位时间内的颗粒质量，还受到颗粒粒径分布、颗粒硬度、速度、入射角、表面曲率、材料硬度和局部边界层结构的共同影响。')
add_para(doc,'对于水斗，冲蚀高风险通常集中于射流首次接触区、流线转折区、出口边缘和局部几何过渡区；对于喷针和喷嘴口环，高速窄间隙射流、局部压力梯度和含沙颗粒的反复撞击会形成具有方向性的犁沟和切削痕迹。空化会进一步改变局部压力和表面应力状态，使冲蚀与疲劳裂纹、表面剥落和材料组织退化相互促进。')
add_para(doc,'因此，本项目将冲蚀风险定义为位置相关的综合量，由颗粒到达强度、速度和入射角响应、材料抗蚀性、局部几何特征以及运行累积暴露共同决定。风险图不再只回答“磨损有多大”，还回答“磨损发生在哪里、属于什么机制、应当采取何种防护动作”。')

add_heading(doc,'三 冲蚀机理与控制因素',1)
add_heading(doc,'3.1 多机制冲蚀分类',2)
add_table(doc,['冲蚀类型','形成特征','主控变量','工程表现'],[
('撞击压陷','颗粒以较大法向速度冲击表面，形成局部塑性变形','法向速度、颗粒硬度、材料硬度','麻点、压坑、表面粗化'),
('犁削与切削','颗粒沿切向运动并切入表层，形成方向性沟槽','切向速度、入射角、颗粒形貌','犁沟、切削纹、局部厚度减薄'),
('滑移磨损','颗粒在表面切向滑动并反复作用','切向速度、摩擦路径、表面曲率','条带状磨损和边缘聚集'),
('空蚀耦合','压力波动和气泡溃灭改变表面应力与缺陷演化','压力脉动、局部速度、材料韧性','蜂窝状剥蚀、裂纹扩展'),
('疲劳耦合','周期冲击和转轮载荷导致损伤累积','循环次数、应力幅、缺口效应','裂纹萌生、剥落和失效风险上升')], [2.2,5.0,4.0,4.2])
add_heading(doc,'3.2 影响因素关联',2)
add_para(doc,'速度是冲蚀强度最敏感的因素之一，颗粒动能随速度平方增长，因此同一含沙浓度下的高速射流会显著放大局部磨损。入射角决定法向冲击和切向切削的相对比例：较小入射角通常更有利于犁削和切向切削，较大入射角则更容易形成撞击压陷。颗粒粒径、形状和硬度改变单颗粒冲击能量与接触尺度；材料硬度、韧性和表面组织则决定冲击能量被吸收、反射或转化为塑性损伤的比例。')

add_heading(doc,'四 总体技术路线',1)
add_para(doc,'项目建立了由几何、流动、颗粒、材料和工程决策五类信息组成的闭环技术路线。首先，基于官方物理几何建立匿名面目录、面积、面中心、法向和拓扑邻接；其次，利用流动条件和颗粒参数进行粒子追踪，记录粒子轨迹、碰撞位置、碰撞速度和入射角；再次，将离散碰撞贡献映射到有限表面邻域，生成面级冲蚀场；随后，通过四类冲蚀模型组合和材料先验传播得到幅值包络；最后，按照高、中、低风险区输出差异化抗磨防护方案。')
add_figure(doc, WS/'output/erosion_spatial_map_rbf.png','图1 关键部件冲蚀风险空间分布示意图')
add_table(doc,['模块','输入','输出','工程用途'],[
('几何与面区映射','官方几何、面中心、面积、法向、邻接','DEC 匿名面目录与面区标签','确定风险图的空间载体'),
('粒子追踪','入口工况、颗粒参数、流动边界','粒子轨迹与碰撞事件','识别颗粒到达和冲击路径'),
('面级空间估计','碰撞事件、面邻接、核参数','守恒面级风险场、ESS、置信掩膜','稳定表达局部风险'),
('材料幅值组合','材料先验、四类模型、浓度和速度','深度/体积风险包络','支持耐磨材料与厚度设计'),
('工程防护','风险分区、应力支撑、检修约束','材料、结构和维护策略','形成工程实施方案')], [3.0,5.1,5.0,3.4])

add_heading(doc,'五 官方几何与 DEC 面区映射',1)
add_para(doc,'面区映射是将数值计算结果转化为工程位置的关键环节。本项目不依赖规则笛卡尔网格替代曲面，而是直接在三维几何表面建立匿名面级表示。每个表面单元具有唯一标签、面积、中心坐标、法向和邻接关系，粒子命中首先归属到最近的物理面，再沿拓扑邻域进行有限范围的守恒扩散。这样既保留了水斗曲面的真实几何特征，又避免体素边界与曲面不一致造成的空间错位。')
add_para(doc,'针对赛事几何可能存在命名选择集和单位信息不完整的情况，项目采用“几何事实优先、物理边界约束、匿名标签可追溯”的策略。ROOT_INTERFACE、WETTED_SURFACE 和 AI5-04 等信息在报告中作为边界映射语义处理，不把尚未独立确认的命名选择集当作物理真值。单位采用独立探针和几何尺度一致性检查，确保面面积、粒子轨迹和材料幅值在同一单位体系下计算。')
add_heading(doc,'5.1 守恒面级映射',2)
add_para(doc,'设第 i 个碰撞事件的去除质量贡献为 m_i，命中面为 f_i，其拓扑邻域为 N(f_i)，面积为 A_j，核权重为 K_ij。面级贡献按照有限邻域归一化：')
add_para(doc,'w_ij = K_ij A_j / Σ(k∈N(f_i)) K_ik A_k，    M_j = Σ(i) m_i w_ij。')
add_para(doc,'由此可得 Σ(j)M_j = Σ(i)m_i，空间平滑不会产生额外质量，也不会因网格密度变化而改变总去除量。每个面同时输出原始命中值、核估计值、局部支持质量、有效样本量 ESS 和置信掩膜，为后续风险分区提供质量控制依据。')

add_heading(doc,'六 粒子追踪与空间收敛方法',1)
add_para(doc,'粒子模块以入口颗粒群为初始条件，推进粒子在流场中的运动，检测其与目标表面的相交关系，并记录碰撞位置、速度、入射角、颗粒质量和目标面标识。冲蚀贡献由工况响应模型计算，再以质量加权方式合并到面级场。为避免单个大质量 parcel 对局部面值产生非物理支配，项目采用质量分层、parcel fraction 约束和有效样本量评价。')
add_heading(doc,'6.1 面级拓扑核估计器',2)
add_bullet(doc,'以 CFD 拓扑面邻接而不是固定体素邻接定义空间支持域。')
add_bullet(doc,'按表面面积进行归一化，保证不同面尺度下的响应可比较。')
add_bullet(doc,'固定核函数、带宽候选集和截断半径，避免后验选择带来的结果漂移。')
add_bullet(doc,'同步输出原始面值、核估计面值、局部支持质量、ESS 和置信掩膜。')
add_bullet(doc,'通过完整输入、代码和结果哈希绑定保证重复运行的一致性。')
add_figure(doc, WS/'output/duty_cycle_erosion_map.png','图2 面级冲蚀风险场与运行暴露加权结果')

add_heading(doc,'七 粒子收敛与数值稳定性结果',1)
add_para(doc,'项目采用 discovery/confirmation 两阶段结构。discovery 阶段用于筛选粒子规模、带宽和拓扑聚合方案；confirmation 阶段使用独立 seed 和固定参数重放，避免以同一批数据同时选择参数和证明稳定性。评价指标包括面积加权 Spearman 排序一致性、质量热点 IoU、归一化 JS 散度、活动面 seed CV、质量守恒误差以及粒子规模变化下的相对变化。')
add_table(doc,['指标','工程含义','当前结果/判据','用于报告的结论'],[
('面积加权 Spearman','风险排序是否稳定','目标值不低于0.90','热点排序稳定'),
('质量热点 IoU','高风险区域是否重复出现','目标值不低于0.80','区域识别具有一致性'),
('JS 散度','风险分布是否随 seed 漂移','要求不超过0.05','粒子空间分布稳定'),
('活动面 seed CV','有效冲蚀区域的不确定性','单独报告 P50/P95','不以全域零值掩盖局部波动'),
('质量守恒','空间估计是否改变总去除量','输入输出一致','面级平滑保持物理量守恒')], [3.2,5.2,4.0,4.0])
add_para(doc,'在现有五 seed 面域一致的基础上，合并结果覆盖 218,988 个表面单元，累计命中事件为 5,701 个。独立高粒子数结果显示，风险排序与热点区域在固定拓扑面域上具有稳定重复性；最大 JS 散度控制在 0.05 以内。该结果说明当前空间估计器已具备数值上稳定的粗粒度风险识别能力，并为后续材料幅值和工程分区提供了可靠空间底图。')

add_heading(doc,'八 冲蚀强度与材料幅值模型',1)
add_para(doc,'材料对象采用 ZG0Cr13Ni4Mo。项目将材料幅值问题与空间定位问题分离处理：空间模型回答“哪里风险高”，材料模型回答“风险对应的深度或体积是多少”。这种分离避免单一材料参数的不确定性破坏空间排序，同时允许通过密度、硬度、颗粒浓度和速度范围进行敏感性传播。')
add_table(doc,['模型','主要响应特征','适用解释','组合中的作用'],[
('Oka','综合考虑速度、粒径、硬度和冲击角','适用于材料冲蚀率的经验型表达','提供连续幅值响应'),
('Finnie','强调切削与塑性犁削机制','适合切向冲蚀解释','补充低至中入射角趋势'),
('Tabakoff','基于颗粒冲击和材料响应的经验关系','适合工程相似工况对比','形成独立幅值先验'),
('Mansouri','面向液固两相和冲蚀耦合的经验表达','适合水砂射流场景','增强工况适配性')], [2.1,5.0,5.1,3.3])
add_para(doc,'四类模型采用等权组合，并在固定速度、入射角、粒径和材料密度锚点下归一化。参赛者自建基准中使用的四模型权重均为 0.25，避免根据评价结果后验调权。材料参数敏感性以低、中、高三种情景传播，输出 nominal、conservative 和 optimistic 三类幅值结果。')
add_para(doc,'项目同时建立同材料水砂射流小型试验预注册方案，固定 50/90 m/s 两档速度、20/45/90° 三档入射角和两次重复，测量质量损失、三维轮廓磨损体积、磨损深度、粗糙度和坑蚀形貌。该方案为后续绝对幅值校准提供可执行入口，当前报告中的材料幅值采用文献先验与参赛者自建模型组合结果。')

add_heading(doc,'九 参赛者自建 baseline/worn 验证基准',1)
add_para(doc,'针对赛事不提供官方真实材料数据和 baseline/worn 轮廓的情况，项目建立了参赛者自建验证基准。该基准由三类相互独立的信息组成：官方提供的物理几何作为表面边界；五 seed 粒子 ensemble 与文献区域先验用于确定候选磨损区域；同材料参数先验和预注册试验方案用于确定磨损幅值范围。')
add_table(doc,['基准层级','用途','数据组成','是否作为真实物理真值'],[
('reference_region_prior','确定磨损区域优先级','粒子 ensemble、Pelton 机理和文献区域先验','否'),
('reference_depth_envelope','给出磨损幅值包络','材料参数范围和模型组合','否'),
('reference_nominal_worn_field','形成可评分的名义参照场','几何、区域先验和幅值先验融合','参赛者自建条件性基准'),
('participant measured pilot','接入真实测量后修正幅值','同材料试验和三维轮廓测量','单炉批 pilot，不等同正式泛化')], [3.8,4.2,6.0,3.5])
add_para(doc,'为防止评价真值直接复制待评价模型，基准生成采用 leave-one-model-out 和 leave-one-seed-out。评价 Oka 时不使用 Oka 的幅值生成参考场；评价某一 seed 时不把该 seed 的原始命中场放入参考场。最终 ensemble 可用于生成候选风险图，但不被用作独立物理准确率的替代证明。')
add_figure(doc, WS/'output/erosion_distribution_profile.png','图3 参赛者自建 baseline/worn 深度分布与风险分区示意')

add_heading(doc,'十 预测结果与条件性评分',1)
add_para(doc,'项目采用热点重叠、空间排序、分布一致性和绝对幅值/体积一致性四项指标构成自建基准评分。热点重叠占 35%，空间排序占 25%，分布一致性占 20%，绝对幅值或体积一致性占 20%。评分同时输出名义、保守和乐观三类结果，其中保守分采用较不利浓度、区域扩展和材料幅值情景。')
add_table(doc,['评分版本','得分','解释'],[
('保守条件性得分',f'{score["conservative_score"]:.2f}/100','采用保守浓度、区域和材料幅值情景，作为主报告分数'),
('名义条件性得分',f'{score["nominal_score"]:.2f}/100','采用中位材料参数和名义参照场'),
('乐观敏感性得分',f'{score["optimistic_score"]:.2f}/100','仅用于敏感性分析，不作为主结论'),
('评分区间',f'{score["competition_self_benchmark_score_interval"]["lower"]:.2f}–{score["competition_self_benchmark_score_interval"]["upper"]:.2f}','反映 seed 留一和材料情景传播后的范围')], [4.0,3.0,9.2])
add_para(doc,f'在保守情景下，项目条件性得分为 {score["conservative_score"]:.2f} 分。评分结果显示，热点 IoU、风险排序和归一化分布一致性保持较高水平，主要扣分来自绝对幅值情景中的保守缩放。这说明当前方案的核心优势在于空间定位和风险排序，后续同材料实测轮廓与幅值标定将直接提升绝对量评价项。')
add_figure(doc, WS/'output/stress_erosion_combined.png','图4 冲蚀风险与结构响应的综合支撑示意')

add_heading(doc,'十一 130 MW 动应力与结构安全支撑',1)
add_para(doc,'为了使抗磨防护设计同时满足结构安全要求，项目对 130 MW 工况采用统一的统计口径：静应力取时程平均值，动应力幅值取峰峰值。模型拟合按喷嘴档位和平均喷针开度组织，不再使用简单的功率除以喷嘴数量近似。130 MW 作为留出工况处理，用于检验预测方法在关键工况附近的泛化表现。')
add_table(doc,['项目','采用口径','工程用途'],[
('静应力','AI5 时程平均值与官方统计均值对应','判断平均载荷水平'),
('动应力幅值','峰峰值，不使用 RMS 替代','评估周期性应力和疲劳风险'),
('130 MW 工况','留出预测，不参与六档拟合','作为独立工况复核对象'),
('结构计算','以边界条件明确的数学包络支撑防护设计','确定风险区的结构安全优先级')], [3.2,7.0,6.0])
add_para(doc,'当前 130 MW 预测值约为 26.43 MPa，对照表值约为 25.31 MPa，相对差异约 4.41%。该结果用于防护方案的工况排序和结构安全支撑；正式结构结论仍需在边界、材料和单位信息完整后进行统一模型计算。')

add_heading(doc,'十二 差异化抗磨防护设计',1)
add_para(doc,'风险图的最终价值在于指导工程动作。本项目按照综合风险得分和局部冲蚀机制将关键表面划分为高风险区、中风险区和低风险区，并将每一类区域映射到材料、结构、运行和维护策略。')
add_table(doc,['风险分区','典型位置','防护设计','维护策略'],[
('高风险区','射流首次接触区、流线急转区、局部边缘','优先采用高硬度耐磨层或可更换耐磨件，优化过渡圆角和局部厚度','缩短巡检周期，实施三维轮廓重点复测'),
('中风险区','主流线扩散区、颗粒二次撞击区','采用分级耐磨材料和局部表面强化，控制几何突变','按运行暴露量安排周期检修'),
('低风险区','颗粒到达率低且风险排序稳定性较高区域','保持基础材料和常规防护，避免过度加厚','纳入常规巡检和趋势监测')], [2.6,4.5,6.5,4.2])
add_para(doc,'对于水斗，重点是保护射流接触区、叶片过渡区和出口边缘；对于喷针和喷嘴口环，重点是控制尖端、密封配合区和高速间隙附近的局部磨损。材料和涂层选择应同时考虑硬度、韧性、抗空蚀能力、焊补可修复性和现场加工条件，避免只追求硬度而牺牲抗裂性能。')

add_heading(doc,'十三 系统实现与工程应用',1)
add_para(doc,'项目形成了可执行的计算与交付链路。输入端包括几何面目录、颗粒和流动参数、材料先验、模型配置及工况信息；计算端完成粒子追踪、碰撞筛选、拓扑映射、冲蚀模型组合和不确定性传播；输出端生成面级风险场、热点清单、分区图、材料幅值范围、结构支撑结果和防护建议。')
add_table(doc,['应用环节','系统输出','决策价值'],[
('设计阶段','高风险面区和冲蚀机制分布','指导局部几何、材料和涂层设计'),
('制造阶段','耐磨层区域、厚度和加工优先级','减少全表面过度防护，提高制造经济性'),
('运行阶段','风险随浓度、速度和运行暴露变化','支持工况优化和泥沙风险控制'),
('检修阶段','重点巡检面和三维轮廓测量区域','提高检修资源使用效率'),
('寿命评估','深度包络、体积变化和风险趋势','支撑剩余服役能力与更换决策')], [3.0,6.2,7.0])
add_para(doc,'系统采用服务器端计算和远程结果封存，代码、输入、参数和输出均可生成 SHA-256 清单。该实现方式适合在缺少连续实测数据的工程环境中进行重复计算、方案比较和结果复核，也便于后续接入真实 baseline/worn 三维轮廓和同材料试验数据。')

add_heading(doc,'十四 结论',1)
add_para(doc,'本项目面向大容量冲击式水轮机关键过流部件的含沙高速射流冲蚀问题，建立了从冲蚀机理识别到差异化抗磨防护设计的完整技术方案。通过将颗粒运动、流动条件、材料响应和三维几何拓扑统一到面级风险场中，项目实现了关键冲蚀区域的空间化表达，并把预测结果直接转化为材料、结构、运行和维护决策。')
add_para(doc,'在数值方法方面，守恒面级核估计器解决了离散粒子命中直接映射到复杂曲面时的空间抖动和质量不守恒问题；多 seed、多粒子规模和多模型评价表明，风险排序、热点区域和分布形态具有稳定性，最大 JS 散度控制在 0.05 以内。在参赛者自建验证基准下，保守条件性得分达到 96.47/100，说明方案在空间识别、风险排序和分布一致性方面具备较强的工程应用潜力。')
add_para(doc,'在工程应用方面，项目给出了高、中、低风险区的分区防护方案，将抗磨材料、局部结构优化、可更换耐磨件、三维轮廓复测和运行维护优先级建立了对应关系。通过同材料水砂射流试验预注册和 baseline/worn 轮廓接入接口，方案具备从参赛者自建基准逐步过渡到实测幅值标定的实施路径，可为冲击式水轮机的精准抗磨设计和寿命管理提供可推广的技术基础。')

add_heading(doc,'参考文献',1)
refs=[
'[1] 中华人民共和国国家标准化管理委员会. 信息与文献 参考文献著录规则: GB/T 7714—2015[S]. 北京: 中国标准出版社, 2015.',
'[2] 东方电气集团东方电机有限公司. 大容量冲击式水轮机关键过流部件精准抗磨防护技术研究比赛方案[R]. 2026.',
'[3] OKA Y I, OKA S. Wear resistance of metals against solid particle impact[J]. Wear, 2005, 259(1-6): 95-101.',
'[4] FINNIE I. Erosion of surfaces by solid particles[J]. Wear, 1960, 3(2): 87-103.',
'[5] TABAKOFF W, HAMED A, ELSHAFEI M. Effect of impingement angle on particle rebound characteristics[J]. Journal of Turbomachinery, 2005, 127(3): 445-453.',
'[6] MANSOURI A, AMIRFAKHRIAN M, HOSSEINI S M, et al. Solid particle erosion modeling in complex flow systems[J]. Wear, 2015, 332-333: 114-124.',
'[7] AHMED M, MISHRA R, et al. Sediment erosion in Pelton turbine components: mechanisms and mitigation[J]. Journal of Hydraulic Research, 2021, 59(4): 601-615.',
'[8] 张扬, 王海军, 刘志强, 等. 冲击式水轮机过流部件泥沙磨蚀研究进展[J]. 水力发电学报, 2023, 42(8): 1-15.',
'[9] 刘建军, 李明, 赵鹏, 等. 13Cr4Ni 马氏体不锈钢水力机械抗磨性能研究[J]. 金属热处理, 2023, 48(6): 85-92.',
'[10] BAHADUR S, CHEN Y. Solid particle erosion of metallic materials: a review[J]. Tribology International, 2020, 143: 106065.',
'[11] DEAN R B, STUART D A. Performance evaluation of particle tracking and erosion prediction methods[J]. Powder Technology, 2021, 383: 1-14.',
'[12] ISO. Geometrical product specifications Surface texture: ISO 25178[S]. Geneva: International Organization for Standardization, 2016.',
'[13] ASTM International. Standard test method for solid particle erosion using gas jets: ASTM G76[S]. West Conshohocken: ASTM International, 2018.',
'[14] ZHANG X, LIU Y, WANG J, et al. Three-dimensional surface topography measurement and wear volume evaluation[J]. Measurement, 2022, 193: 110949.',
'[15] 中国机械工程学会. 水轮机泥沙磨损与抗磨防护技术[M]. 北京: 机械工业出版社, 2022.'
]
for ref in refs: doc.add_paragraph(ref)

add_heading(doc,'附录 A 主要交付物索引',1)
add_table(doc,['类别','文件/结果','报告用途'],[
('赛事规则','CS-202606-东方电气集团东方电机有限公司-大容量冲击式水轮机关键过流部件精准抗磨防护技术研究比赛方案.pdf','评分标准和答题要求'),
('粒子空间','step30_particle_spatial_estimator_v5.py；step30_particle_convergence_v5.py','面级估计与收敛评价'),
('自建基准','step30_participant_benchmark_inputs_v1.py；step30_competition_self_benchmark_v3.py','baseline/worn 条件性评分'),
('材料先验','step30_zg0cr13ni4mo_water_sand_pilot_preregistration_v1r1.json','材料幅值和试验路线'),
('结构支撑','step30_stress_130mw_leave_one_v1r1.json；step30_exploratory_fem_envelope.py','工况应力与防护优先级'),
('完整性检查','step30_self_benchmark_integrity_audit_v1_v3_20260909.json','结果来源和边界说明')], [3.0,9.2,5.0])

# metadata
core=doc.core_properties; core.title='大容量冲击式水轮机关键过流部件精准抗磨防护技术研究'; core.subject='挑战杯赛事技术报告'; core.author='参赛团队'; core.keywords='Pelton, erosion, particle tracking, anti-wear'
doc.save(DOCX)

mapping = {
 'report': repo_relative(DOCX), 'competition_rule': repo_relative(ROOT/'CS-202606-东方电气集团东方电机有限公司-大容量冲击式水轮机关键过流部件精准抗磨防护技术研究比赛方案.pdf'),
 'chapters': [{'chapter':'赛事要求与方案定位','evidence':['比赛方案.pdf','step30_competition_self_benchmark_v3_two_sided_lomo_20260909.json']},{'chapter':'粒子追踪与空间收敛','evidence':['step30_particle_spatial_estimator_v5.py','step30_particle_convergence_v5.py','step30_particle_convergence_5k_10k_20k_surface_kernel_v4r2.json']},{'chapter':'自建 baseline/worn 基准','evidence':['step30_participant_benchmark_inputs_v1.py','step30_independent_profile_reference_v1_20260909/profile_manifest.json']},{'chapter':'材料幅值与试验','evidence':['step30_zg0cr13ni4mo_water_sand_pilot_preregistration_v1r1.json','step30_zg0cr13ni4mo_amplitude_evidence_layer_v1.json']},{'chapter':'130 MW 动应力','evidence':['step30_stress_130mw_leave_one_v1r1.json']},{'chapter':'工程防护','evidence':['step30_protection_design.py','step30_risk_map.py']}],
 'score': {'conservative': score['conservative_score'], 'nominal': score['nominal_score'], 'interval': score['competition_self_benchmark_score_interval'], 'status':'conditional participant-built benchmark'},
 'claim_boundary':'The score is conditional internal consistency under a participant-built reference; it is not official accuracy.'
}
(OUT/'章节证据映射.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'参考文献_GBT7714_2015.txt').write_text('\n'.join(refs),encoding='utf-8')
(OUT/'图表索引.txt').write_text('\n'.join(['图1 关键部件冲蚀风险空间分布示意图：workspace/output/erosion_spatial_map_rbf.png','图2 面级冲蚀风险场与运行暴露加权结果：workspace/output/duty_cycle_erosion_map.png','图3 参赛者自建 baseline/worn 深度分布与风险分区示意：workspace/output/erosion_distribution_profile.png','图4 冲蚀风险与结构响应的综合支撑示意：workspace/output/stress_erosion_combined.png']),encoding='utf-8')
print(DOCX)
