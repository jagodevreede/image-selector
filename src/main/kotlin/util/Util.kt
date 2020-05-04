package util

object Util {
    fun toReadableTime(ms: Long): String {
        val milliseconds = (ms % 1000 / 100).toInt()
        val seconds = (ms / 1000 % 60).toInt()
        val minutes = (ms / (1000 * 60) % 60).toInt()
        val hours = (ms / (1000 * 60 * 60) % 24).toInt()

        return "%02d:%02d:%02d.%d".format(hours, minutes, seconds, milliseconds)
    }

}