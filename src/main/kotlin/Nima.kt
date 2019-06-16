package main.kotlin

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.io.FileReader


class Nima(folder: File) : AutoCloseable {
    private val ratingsByFolder = mutableMapOf<File, List<PredictionOutcome>>()
    private val predictionOutcomeType = object : TypeToken<List<PredictionOutcome>>() {}.type

    fun run(file: File): Double {
        return getRating(file).filter { it.image_id == file.name.substring(0, file.name.length - 4) }
                        .map { it.mean_score_prediction }
                        .getOrElse(0) { 0.toDouble() }
    }

    private fun getRating(file: File): List<PredictionOutcome> {
        return ratingsByFolder.getOrPut(file.parentFile, {
            Gson().fromJson<List<PredictionOutcome>>(FileReader(File(file.parentFile, "rating.json")), predictionOutcomeType)
        })
    }

    override fun close() {

    }

}