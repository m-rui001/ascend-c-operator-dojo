/*
 * Round 36 - 我实现的 TopKV3 结构复述 mini（proposal 打包 → 排序 → MrgSort4 归并 → 解码）
 */
#include "kernel_operator.h"

constexpr uint32_t PROPOSAL_NUM_PER_REP = 16;

class MyTopK {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR tiling)
    {
        this->numCol = tdNumCol;
        this->kValue = tdK;
        pipe.InitBuffer(inQueueX, 2, rowAlign * sizeof(half));
        pipe.InitBuffer(idxPredicateBuf, 1, rowAlign * sizeof(int32_t));   // 基础索引向量
        pipe.InitBuffer(proposalBuf, 1, rowAlign * 2 * sizeof(half));      // 打包 proposal
        pipe.InitBuffer(topkPingpong, 2, kValue * PROPOSAL_NUM_PER_REP * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // ---- 基础索引向量（新 API：CreateVecIndex）----
        AscendC::LocalTensor<int32_t> baseIdx = idxPredicateBuf.Get<int32_t>();
        AscendC::CreateVecIndex(baseIdx, 0, rowAlign);
        AscendC::PipeBarrier<AscendC::PIPE_V>();

        uint32_t jRep = (numCol + rowAlign - 1) / rowAlign;
        for (uint32_t j = 0; j < jRep; j++) {
            uint32_t cnt = (j == jRep - 1) ? colTail : rowAlign;
            // 载入分块 + 取负（最小化场景）
            AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
            AscendC::DataCopy(x, xGm[(uint64_t)j * rowAlign], rowAlign);
            inQueueX.EnQue(x);
            x = inQueueX.DeQue<half>();
            if (!largest) {
                AscendC::Muls(x, x, (half)(-1.0), cnt);
                AscendC::PipeBarrier<AscendC::PIPE_V>();
            }
            // pad 负无穷到 16 倍数（少量填充，标量正当）
            uint32_t padCnt = cnt % PROPOSAL_NUM_PER_REP;
            if (padCnt) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                for (uint32_t n = 0; n < padCnt; n++) {
                    x.SetValue(cnt + n, FP16_NEG_INF);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            }
            // ---- proposal 编码：score 与 index 打包（//?: 打包指令细节按生产 ProposalConcat 形态）----
            AscendC::LocalTensor<half> prop = proposalBuf.Get<half>();
            ProposalConcat(prop, x, cnt / PROPOSAL_NUM_PER_REP, 4);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            // ---- 块内排序（硬件排序指令，//?: 指令名随版本，MrgSort4 前置排序见生产）----
            // ---- MrgSort4 四路归并（ping-pong）----
            MrgSortCustom(j, prop);
            inQueueX.FreeTensor(x);
        }
        // ---- 解码：values = proposal 高半，indices = 低半（//?: 拆包形态按生产 Decode）----
        // CopyOutValues / CopyOutIndices 省略
    }

private:
    __aicore__ inline void MrgSortCustom(uint32_t round, AscendC::LocalTensor<half>& prop)
    {
        AscendC::LocalTensor<half> dst = topkPingpong.Get<half>()[pingpong * kHalf];
        if (round == 0) {
            // 单块：自身为四同源归并（复制为四路）
            AscendC::MrgSortSrcList<half> srcList(prop, prop, prop, prop);
            AscendC::MrgSort4Info info({(uint32_t)PROPOSAL_NUM_PER_REP, (uint32_t)PROPOSAL_NUM_PER_REP,
                                        (uint32_t)PROPOSAL_NUM_PER_REP, (uint32_t)PROPOSAL_NUM_PER_REP},
                                       true, 1, 1);
            AscendC::MrgSort4(dst, srcList, info);
        } else {
            // 归并：上一轮结果 + 新块
            AscendC::LocalTensor<half> prev = topkPingpong.Get<half>()[!pingpong * kHalf];
            AscendC::MrgSortSrcList<half> srcList(prev, prev, prop, prop);
            AscendC::MrgSort4Info info({(uint32_t)kValue, (uint32_t)kValue,
                                        (uint32_t)PROPOSAL_NUM_PER_REP, (uint32_t)PROPOSAL_NUM_PER_REP},
                                       true, 15, 1);  // k<16 时可提前暂停耗尽源
            AscendC::MrgSort4(dst, srcList, info);
        }
        pingpong = !pingpong;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> idxPredicateBuf, proposalBuf, topkPingpong;
    AscendC::GlobalTensor<half> xGm, valGm, idxGm;
    uint32_t numCol, kValue, rowAlign = 128, colTail;
    bool largest = true;
    bool pingpong = false;
    static constexpr uint32_t kHalf = 64;
    static constexpr half FP16_NEG_INF = -65504.0_h;
};
