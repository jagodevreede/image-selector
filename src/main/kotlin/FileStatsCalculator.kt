import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.objdetect.Objdetect
import util.CV.loadMat
import util.CV.resizeToSameSize
import util.Util.toReadableTime
import java.io.File
import java.io.FileReader
import java.io.FileWriter
import java.nio.file.Files
import java.nio.file.attribute.BasicFileAttributes
import java.time.LocalDateTime
import java.time.ZoneId
import java.util.*

class FileStatsCalculator(private val folder: File, private val nima: Nima) {

    private val faceCascade = CascadeClassifier("lbpcascade_frontalface.xml")
    private val statsList = mutableListOf<FileStats>()
    private val gson = GsonBuilder().setPrettyPrinting().create()
    private val fileStatsType = object : TypeToken<List<FileStats>>() {}.type

    fun calulate(): List<FileStats> {
        val statsFile = File(folder, "file_stats.json")
        if (statsFile.exists()) {
            statsList.addAll(Gson().fromJson<List<FileStats>>(FileReader(statsFile), fileStatsType))
        } else {
            val startTime = System.currentTimeMillis()
            processFolder(folder)
            println("Total calculation done in: " + toReadableTime(System.currentTimeMillis() - startTime))
            FileWriter(statsFile).use { writer ->
                gson.toJson(statsList, writer)
                writer.flush()
            }

        }
        return statsList.toList()
    }

    private fun processFolder(processingFolder: File) {
        processingFolder.walkTopDown().forEachParallel {
            if (it.isFile && it.canRead() && isJpeg(it)) {
                try {
                    processFile(it)
                } catch (e: Exception) {
                    println(it.absolutePath + " => Failed to load image file due to: " + e.message)
                    throw e
                }
            }
        }
    }

    fun <A>Sequence<A>.forEachParallel(f: suspend (A) -> Unit): Unit = runBlocking {
        map { async(Dispatchers.IO) { f(it) } }.forEach { it.await() }
    }

    private fun isJpeg(it: File): Boolean {
        return it.name.matches(Regex(".*\\.jpg$"))
    }

    private fun processFile(it: File) {
        val startTime = System.currentTimeMillis()

        val rating = nima.run(it)

        val mat = loadMat(it)
        val matGray = Mat()
        Imgproc.cvtColor(mat, matGray, Imgproc.COLOR_RGB2GRAY)

        val stat = FileStats(it, rating, calcLaplician(matGray).toInt(), countFaces(matGray), getCreateTimeStamp(it), mat.size(), getHistogramData(mat))
        statsList.add(stat)

        println(stat.toString() + " in %4dms ".format(System.currentTimeMillis() - startTime) + Thread.currentThread().name)

    }

    private fun getCreateTimeStamp(it: File): LocalDateTime {
        val attr = Files.readAttributes(it.toPath(), BasicFileAttributes::class.java)
        return LocalDateTime.ofInstant(attr.lastModifiedTime().toInstant(), ZoneId.systemDefault())
    }

    private var ranges = MatOfFloat(0f, 256f)
    private var histSize = MatOfInt(128) // 25

    private fun getHistogramData(img: Mat): List<Double> {
        val hist = Mat()

        Imgproc.calcHist(Arrays.asList(img), MatOfInt(0),
                Mat(), hist, histSize, ranges)

        val data = mutableListOf<Double>()

        for (row in 0 until hist.rows()) {
            val value = hist.get(row, 0)[0]
            data.add(value)
        }
        return data.toList()
    }

    private fun countFaces(matGray: Mat): Int {
        val height = matGray.rows()
        val absoluteFaceSize: Double = Math.round(height * 0.2f).toDouble()

        val faces = MatOfRect()

        faceCascade.detectMultiScale(matGray, faces, 1.1, 2, 0 or Objdetect.CASCADE_SCALE_IMAGE, Size(absoluteFaceSize, absoluteFaceSize), Size())

        return faces.toArray().size
    }

    private fun calcLaplician(matGray: Mat): Double {
        val resizedImage = resizeToSameSize(matGray)

        val destination = Mat()
        Imgproc.Laplacian(resizedImage, destination, 3)
        val std = MatOfDouble()
        Core.meanStdDev(destination, MatOfDouble(), std)

        return Math.pow(std.get(0, 0)[0], 2.0)
    }

}

data class FileStats(
        val file: File,
        val rating: Double,
        val sharpness: Int,
        val faces: Int,
        val dateTaken: LocalDateTime,
        val size: Size,
        val histogramData: List<Double>,
        val ignored: Boolean = false
) {
    fun ratingWeighted(maxSharpness: Double): Double = this.rating + ((this.rating * (this.sharpness / maxSharpness) * Runner.SHARPNESS_MULTIPLIER))

    override fun toString(): String {
        return "%35s => %4d there are %2d faces rating: %.2f taken @".format(file.name, sharpness, faces, rating) + dateTaken
    }
}