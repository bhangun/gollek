package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.Renderer;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import javax.imageio.ImageIO;

/**
 * Rich display rendering for Gollek types in Jupyter.
 * Falls back to plain text for unknown types.
 */
public class TensorDisplay {

    private TensorDisplay() {}

    /**
     * Returns a {@link DisplayData} with HTML + plain text for known Gollek
     * types, or {@code null} to let the caller fall back to {@code toString()}.
     */
    public static DisplayData render(Object value, Renderer renderer) {
        if (value == null) return null;

        // Use reflection so this compiles even if gollek-ml-tensor is absent at compile time
        try {
            Class<?> tensorClass = Class.forName("tech.kayys.gollek.spi.tensor.Tensor");
            if (tensorClass.isInstance(value)) {
                return renderTensor(value, tensorClass);
            }
        } catch (ClassNotFoundException ignored) {}

        return null;
    }

    private static DisplayData renderTensor(Object tensor, Class<?> cls) {
        try {
            long[] shape = (long[]) cls.getMethod("shape").invoke(tensor);
            String device = (String) cls.getMethod("device").invoke(tensor);
            String dtype = cls.getMethod("dtype").invoke(tensor).toString();

            String html = buildHtml(tensor, cls, shape, device, dtype);
            String plain = String.format("Tensor(shape=%s, device=%s, dtype=%s)",
                    java.util.Arrays.toString(shape), device, dtype);

            DisplayData data = new DisplayData(plain);
            data.putData("text/html", html);
            return data;
        } catch (Exception e) {
            return null;
        }
    }

    private static String buildHtml(Object tensor, Class<?> cls,
                                    long[] shape, String device, String dtype) {
        StringBuilder sb = new StringBuilder();
        sb.append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>");
        sb.append("<b>Tensor</b> shape=").append(java.util.Arrays.toString(shape))
          .append(" device=").append(device).append(" dtype=").append(dtype);

        // Heatmap for 2-D tensors up to 32×32
        if (shape.length == 2 && shape[0] <= 32 && shape[1] <= 32) {
            String img = heatmap(tensor, cls, (int) shape[0], (int) shape[1]);
            if (img != null) {
                sb.append("<br><img src='data:image/png;base64,").append(img)
                  .append("' style='image-rendering:pixelated;max-width:256px'/>");
            }
        }
        sb.append("</div>");
        return sb.toString();
    }

    private static String heatmap(Object tensor, Class<?> cls, int rows, int cols) {
        try {
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            double[][] data = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double v = (double) cls.getMethod("getDouble", int.class, int.class)
                                          .invoke(tensor, i, j);
                    data[i][j] = v;
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
            }
            double range = max - min == 0 ? 1 : max - min;
            BufferedImage img = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) {
                    float hue = 0.6f * (1f - (float) ((data[i][j] - min) / range));
                    img.setRGB(j, i, Color.HSBtoRGB(hue, 1f, 1f));
                }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            ImageIO.write(img, "png", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (Exception e) {
            return null;
        }
    }
}
