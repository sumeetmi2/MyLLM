package com.myllm.dao;

import static com.myllm.utils.TestUtils.assertMatrixEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class MatrixTest {

    @Test
    @DisplayName("matrix multiplication")
    void multiply() throws Exception {
        Matrix m1 = Matrix.of(new double[][] {{1, 1, 3}, {1, 1, 1}});
        Matrix m2 = Matrix.of(new double[][] {{1, 1}, {1, 1}, {1, 1}});
        Matrix expected = Matrix.of(new double[][] {{5, 5}, {3, 3}});
        assertEquals(expected, m1.multiply(m2));
    }

    @Test
    @DisplayName("Soft Max Test")
    void softMax() {
        Matrix m1 = Matrix.of(new double[][] {{2.0, 1.0, 0.1}, {1.0, 5.0, 2.0}});
        double[][] output = {
            {0.6590011388859679, 0.24243297070471392, 0.09856589040931818},
            {0.01714782554552039, 0.9362395518765058, 0.0466126225779739}
        };
        // float keeps ~7 significant digits
        assertMatrixEquals(output, m1.softMax(), 1e-6);
    }

    @Test
    @DisplayName("ReLU")
    void relu() {
        Matrix m1 = Matrix.of(new double[][] {{2.0, -1.5, 0.0}, {-0.5, 4.3, -12.1}});
        Matrix result = Matrix.of(new double[][] {{2.0, 0.0, 0.0}, {0.0, 4.3, 0.0}});
        assertEquals(result, m1.relu());
    }

    @Test
    @DisplayName("data length must equal rows x cols")
    void rejectsWrongLength() {
        assertThrows(IllegalArgumentException.class, () -> new Matrix(2, 3, new float[5]));
        assertThrows(IllegalArgumentException.class, () -> new Matrix(0, 3, new float[0]));
    }

    @Test
    @DisplayName("get, set and offset address data row-major")
    void rowMajorIndexing() {
        Matrix m = Matrix.of(new double[][] {{1, 2, 3}, {4, 5, 6}});
        assertEquals(6.0f, m.get(1, 2));
        assertEquals(3, m.offset(1));
        assertEquals(4.0f, m.data()[m.offset(1)]);
        m.set(0, 1, 9.0f);
        assertEquals(9.0f, m.data()[1]);
    }
}
