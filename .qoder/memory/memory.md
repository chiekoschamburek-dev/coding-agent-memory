
## [2026-10-02] 任务:调研 2026/2025 LLM agent 长期记忆的召回后排序(post-recall ranking)方法
- 产出:按系统整理 MemX(2603.16171 四因子 0.45/0.25/0.05/0.10 + 半衰期30天 + RRF k=60)、CoEvo-Mem/SR-QR(2608.01739 路由加权RRF+λQ=0.15 utility rank, ablation -4.24/-4.80)、VikingMem(2605.29640 S_final 含 time/business 权重, rerank -3.64)、HyMem(2602.13933)、LightMem-SLM(2604.07798 2K→K SLM 过滤)、HORMA(2606.11680)、Zep/Mem0/A-MEM/MemoryOS/MIRIX/LightMem/Memory-R1/HippoRAG2 等。
- 结论要点:混合召回+RRF(k=60)或多因子加权和是主流;cross-encoder 只在 20-50 候选阶段用;RankGPT 式 listwise 重排在记忆系统里几乎缺席;Memory-R1 报告重排收益小延迟大。
- 工具经验:arXiv HTML 用 curl 抓取后正则去标签再 grep,比 WebFetch 摘要更精确(gx.py 在 D:/tmp)。
