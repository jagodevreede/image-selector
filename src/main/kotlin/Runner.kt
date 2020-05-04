import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import util.Util.toReadableTime
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.time.Duration
import java.time.ZoneOffset
import java.time.temporal.ChronoUnit


class Runner(private val folder: File, private val outputFolder: File, private val nima: Nima, private val reduceTo: Int, private val similarThreshold: Double) {

    companion object{
        const val AVG_DURATION_MULTIPLIER = 2
        const val SHARPNESS_MULTIPLIER = 0.1
    }

    private val collectionToKeep = mutableSetOf<FileStats>()
    private lateinit var rawFolderStats: MutableList<FileStats>

    private fun processFolderFileStats(inComingFileStats: List<FileStats>): MutableList<FileStats> {
        println("Using similarThreshold: $similarThreshold")
        collectionToKeep.clear()
        val fileStats = inComingFileStats.filter { !it.ignored }
        val similarImagesByFileStat = findSimilarImages(fileStats)

        println("Processing %d files to %d".format(fileStats.count(), reduceTo))
        val orderedByDate = fileStats.sortedBy { it.dateTaken }

        val avgDuration = findAvgBetweenPhotos(orderedByDate)
        println("Avg between photo's %d seconds".format(avgDuration))

        val collections: MutableList<List<FileStats>> = findCollectionsByDateRoot(orderedByDate, avgDuration)

        val totalRemainingCollection = mutableSetOf<FileStats>()

        val averageSharpness = fileStats.sortedBy { it.sharpness }.map { it.sharpness }.average()
        val maxSharpness: Int = fileStats.sortedBy { it.sharpness }.map { it.sharpness }.max()!!
        val averageRating = fileStats.sortedBy { it.rating }.map { it.rating }.average()

        collections.forEach { fileStatsList ->
            val perCollection = Math.max((reduceTo.toDouble() / fileStats.count() * fileStatsList.count()).toInt(), 1)
            val sorted: List<FileStats> = fileStatsList.filter { it.sharpness > averageSharpness / 1.5 }.sortedBy { it.rating }

            println("    collection of ${fileStatsList.count()} reduced to ${sorted.count()} sharp photos will take $perCollection")

            val remainingCollection: MutableList<FileStats> = sorted.toMutableList()
            val toKeep = getImagesToKeep(remainingCollection, perCollection, similarImagesByFileStat)

            collectionToKeep.addAll(toKeep)
            totalRemainingCollection.addAll(remainingCollection)

            println("    %d take max %d -> %d".format(fileStatsList.count(), perCollection, toKeep.size))
        }

        println("Filling with %d".format(reduceTo - collectionToKeep.count()))

        val remaining = totalRemainingCollection.toMutableList()
        remaining.sortBy { it.rating }

        val toKeep = getImagesToKeep(remaining.filter { it.sharpness > averageSharpness }.toMutableList(), reduceTo - collectionToKeep.count(), similarImagesByFileStat)

        collectionToKeep.addAll(toKeep)

        println("photo's to keep: ${collectionToKeep.size}")
        collectionToKeep.sortedBy { it.file }.forEach {
            println(it.toString() + " ratingWeighted " + it.ratingWeighted(maxSharpness.toDouble()))
        }
        println("Average rating was    $averageRating final collection: ${collectionToKeep.map { it.rating }.average()}")
        println("Average sharpness was $averageSharpness final collection: ${collectionToKeep.map { it.sharpness }.average()}")
        println("Max     sharpness     $maxSharpness final collection: ${collectionToKeep.map { it.sharpness }.max()}")

        return collectionToKeep.sortedBy { it.file.name }.toMutableList()
    }

    fun getResult(): MutableList<FileStats> {
        rawFolderStats = FileStatsCalculator(folder, nima).calulate().toMutableList()

        return processFolderFileStats(rawFolderStats)
    }

    fun updateResults(fileStats: List<FileStats>): MutableList<FileStats> {
        fileStats.filter { it.ignored }.forEach { newStat ->
            rawFolderStats.removeAll { it.file.absolutePath == newStat.file.absolutePath }
        }

        return processFolderFileStats(rawFolderStats)
    }

    fun copyToOutput() {
        collectionToKeep.sortedBy { it.file }.forEach {
            copyFile(it.file, File(outputFolder, it.file.name))
        }
    }

    private fun findCollectionsByDateRoot(orderedByDate: List<FileStats>, avgDuration: Long): MutableList<List<FileStats>> {
        var collections: MutableList<List<FileStats>>
        var reduceTimes = 0
        do {
            reduceTimes++
            collections = findCollectionsByDate(orderedByDate, avgDuration * reduceTimes)

            println("Found %d collections".format(collections.size))
        } while (collections.count() > (reduceTo /2))
        return collections
    }

    private fun getImagesToKeep(remainingCollection: MutableList<FileStats>, maxToKeep: Int, similarImagesByFileStat: Map<FileStats, List<FileStats>>): MutableList<FileStats> {
        val toKeep = mutableListOf<FileStats>()

        while (remainingCollection.isNotEmpty() && maxToKeep > toKeep.size) {
            val best: FileStats = remainingCollection[0]
            println("  Keep ${best.file.name} out of ${remainingCollection.count()}")
            toKeep.add(best)
            remainingCollection.remove(best)
            val similar: List<FileStats> = similarImagesByFileStat[best].orEmpty()
            if (similar.count() > 0) {
                remainingCollection.removeAll(similar)
                println("      Removing ${similar.count()} similar photos as ${best.file.name} now there are ${remainingCollection.count()} left")
            }
            val sameTime: List<FileStats> = remainingCollection.filter { Math.abs(Duration.between(it.dateTaken, best.dateTaken).seconds) < 90 }
            if (sameTime.count() > 0) {
                remainingCollection.removeAll(sameTime)
                println("      Removing ${sameTime.count()} photos around the same time as ${best.file.name} now there are ${remainingCollection.count()} left")
            }
        }
        return toKeep
    }

    private fun findSimilarImages(orderedByDate: List<FileStats>): Map<FileStats, List<FileStats>> {
        val histogramByFileStat = calculateHistograms(orderedByDate)

        val startTime = System.currentTimeMillis()
        println("Finding similar images...")

        val similarImagesByFileStat = orderedByDate.associateBy({ it }, { stat ->
            val ownHistogram = histogramByFileStat.get(stat)
            orderedByDate.filter { it.file != stat.file }.filter {
                val histogram = histogramByFileStat.get(it)
                val res: Double = Imgproc.compareHist(ownHistogram, histogram, Imgproc.CV_COMP_CORREL)
                //println(stat.file.name + " similar to " + it.file.name + " " + res)

                res > similarThreshold
            }
        })
        println("Finding similar images done in %s".format(toReadableTime(System.currentTimeMillis() - startTime)))
        return similarImagesByFileStat
    }

    private fun calculateHistograms(fileStats: List<FileStats>): Map<FileStats, Mat> {
        println("Calculate histograms...")
        val startTime = System.currentTimeMillis()

        val chunks: List<List<FileStats>> = fileStats.chunked(fileStats.count() / 8)
        val result = mutableMapOf<FileStats, Mat>()
        chunks.forEachParallel { fileStatsChunk ->
            val histograms = fileStatsChunk.associateBy({ it }, {
                val data = it.histogramData
                val restoreMat = Mat(data.size, 1, 5)

                data.forEachIndexed { index, d ->
                  restoreMat.put(index, 0, d)
                }
                restoreMat
            })
            synchronized(result) {
                result.putAll(histograms)
            }
        }

        println("Calculate histograms done in %s".format(toReadableTime(System.currentTimeMillis() - startTime)))
        return result
    }


    fun <A>Collection<A>.forEachParallel(f: suspend (A) -> Unit): Unit = runBlocking {
        map { async(Dispatchers.IO) { f(it) } }.forEach { it.await() }
    }

    @Throws(IOException::class)
    private fun copyFile(source: File, dest: File) {
        FileInputStream(source).channel.use { sourceChannel ->
            FileOutputStream(dest).channel.use { destChannel ->
                destChannel.transferFrom(sourceChannel, 0, sourceChannel!!.size())
            }
        }
    }

    private fun findAvgBetweenPhotos(orderedByDate: List<FileStats>): Long {
        val firstDate = orderedByDate.first().dateTaken.toEpochSecond(ZoneOffset.UTC)
        val lastDate = orderedByDate.last().dateTaken.toEpochSecond(ZoneOffset.UTC)
        return (lastDate - firstDate) / orderedByDate.count()
    }

    private fun findCollectionsByDate(orderedByDate: List<FileStats>, avgDuration: Long): MutableList<List<FileStats>> {
        val collections = mutableListOf<List<FileStats>>()
        var collection = mutableListOf<FileStats>()
        var lastElementInCollection = orderedByDate.first()
        collections.add(collection)
        orderedByDate.forEach {
            if (lastElementInCollection.dateTaken.until(it.dateTaken, ChronoUnit.SECONDS) > avgDuration * AVG_DURATION_MULTIPLIER && collection.size > reduceTo / 10) {
                collection = mutableListOf()
                collections.add(collection)
            }
            collection.add(it)
            lastElementInCollection = it
        }
        return collections
    }
}
