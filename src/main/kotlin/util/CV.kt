package main.kotlin

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import javax.imageio.ImageIO

object CV {
    private fun loadBufferedImage(file: File): BufferedImage {
        return ImageIO.read(file)
    }

    fun bufferedImageToMat(bi: BufferedImage): Mat {
        val mat = Mat(bi.height, bi.width, CvType.CV_8UC3)
        val data = (bi.raster.dataBuffer as DataBufferByte).data
        mat.put(0, 0, data)
        return mat
    }

    fun loadMat(file: File): Mat {
        return bufferedImageToMat(loadBufferedImage(file))
    }

    fun resizeToSameSize(matGray: Mat): Mat {
        val resizedImage = Mat()
        val sz = Size(1024.toDouble(), 1024.toDouble())
        Imgproc.resize(matGray, resizedImage, sz)
        return resizedImage
    }
}