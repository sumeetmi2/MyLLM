package com.myllm.batched;

import com.myllm.attention.AttentionHead;
import com.myllm.dao.Matrix;
import com.myllm.lora.LoraAdapter;
import java.util.stream.IntStream;

/**
 * One attention head over stacked sequences.
 *
 * <p>The Q/K/V projections (and their gradients) are single matmuls over every row of the batch; scores, causal
 * mask and softmax run per sequence, so no sequence attends to another. Same math as {@code AttentionHead}, which
 * stays the readable single-sequence version; this class keeps its own caches.
 */
public class BatchedAttentionHead {

    private final AttentionHead head;
    private final int d_k;
    private Matrix q;
    private Matrix k;
    private Matrix v;
    // softmax weights of each sequence, len x len
    private Matrix[] attnWeights;
    private int[] lengths;
    private int[] starts;

    public BatchedAttentionHead(AttentionHead head) {
        this.head = head;
        this.d_k = head.getWQ().cols();
    }

    public Matrix forwardPass(Matrix x, int[] lengths) {
        LoraAdapter loraQ = head.getLoraQ();
        LoraAdapter loraV = head.getLoraV();
        q = x.multiply(head.getWQ());
        if (loraQ != null) {
            q = q.add(loraQ.forwardPass(x));
        }
        k = x.multiply(head.getWK());
        v = x.multiply(head.getWV());
        if (loraV != null) {
            v = v.add(loraV.forwardPass(x));
        }
        this.lengths = lengths;
        this.starts = BatchOps.starts(lengths);
        attnWeights = new Matrix[lengths.length];
        Matrix out = new Matrix(x.rows(), d_k);
        // sequences are independent; each task writes only its own rows of out
        IntStream.range(0, lengths.length).parallel().forEach(s -> {
            Matrix qs = BatchOps.rows(q, starts[s], lengths[s]);
            Matrix ks = BatchOps.rows(k, starts[s], lengths[s]);
            Matrix vs = BatchOps.rows(v, starts[s], lengths[s]);
            Matrix scores = qs.multiplyTranspose(ks).scale(1 / Math.sqrt(d_k));
            attnWeights[s] = scores.applyCausalMask().softMax();
            BatchOps.writeRows(out, starts[s], attnWeights[s].multiply(vs));
        });
        return out;
    }

    public AttentionHead.AttentionBackwardResult backwardPass(Matrix x, Matrix dOutput) {
        Matrix dQ = new Matrix(q.rows(), d_k);
        Matrix dK = new Matrix(k.rows(), d_k);
        Matrix dV = new Matrix(v.rows(), d_k);
        IntStream.range(0, lengths.length).parallel().forEach(s -> {
            Matrix qs = BatchOps.rows(q, starts[s], lengths[s]);
            Matrix ks = BatchOps.rows(k, starts[s], lengths[s]);
            Matrix vs = BatchOps.rows(v, starts[s], lengths[s]);
            Matrix dOut = BatchOps.rows(dOutput, starts[s], lengths[s]);
            Matrix attn = attnWeights[s];
            Matrix dAttnWeights = dOut.multiplyTranspose(vs);
            BatchOps.writeRows(dV, starts[s], attn.transposeMultiply(dOut));
            Matrix dQKT = softmaxBackward(attn, dAttnWeights).scale(1 / Math.sqrt(d_k));
            BatchOps.writeRows(dQ, starts[s], dQKT.multiply(ks));
            BatchOps.writeRows(dK, starts[s], dQKT.transposeMultiply(qs));
        });
        // weight gradients sum over every sequence in one matmul each
        Matrix dWq = x.transposeMultiply(dQ);
        Matrix dWk = x.transposeMultiply(dK);
        Matrix dWv = x.transposeMultiply(dV);
        Matrix dX_FromQ = dQ.multiplyTranspose(head.getWQ());
        Matrix dX_FromK = dK.multiplyTranspose(head.getWK());
        Matrix dX_FromV = dV.multiplyTranspose(head.getWV());

        LoraAdapter loraQ = head.getLoraQ();
        LoraAdapter loraV = head.getLoraV();
        LoraAdapter.LoraAdapterBackwardPassResult loraQResult = null;
        LoraAdapter.LoraAdapterBackwardPassResult loraVResult = null;
        Matrix dX = dX_FromK.add(dX_FromQ).add(dX_FromV);
        if (loraQ != null) {
            loraQResult = loraQ.backwardPass(x, dQ);
            dX = dX.add(loraQResult.dX());
        }
        if (loraV != null) {
            loraVResult = loraV.backwardPass(x, dV);
            dX = dX.add(loraVResult.dX());
        }
        return new AttentionHead.AttentionBackwardResult(dX, dWq, dWk, dWv, loraQResult, loraVResult);
    }

    // dScores[i][j] = s[i][j] * (g[i][j] - g[i]·s[i]), as in AttentionHead
    private static Matrix softmaxBackward(Matrix attn, Matrix dAttnWeights) {
        int rows = dAttnWeights.rows();
        int cols = dAttnWeights.cols();
        float[] gData = dAttnWeights.data();
        float[] sData = attn.data();
        Matrix diffMatrix = dAttnWeights.copy();
        float[] diff = diffMatrix.data();
        for (int i = 0; i < rows; i++) {
            int start = i * cols;
            double dot = 0;
            for (int j = 0; j < cols; j++) {
                dot += gData[start + j] * sData[start + j];
            }
            for (int j = 0; j < cols; j++) {
                diff[start + j] = (float) (diff[start + j] - dot);
            }
        }
        return attn.elementWiseOp(diffMatrix, (s, g) -> s * g);
    }
}
