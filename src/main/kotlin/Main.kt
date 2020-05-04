import java.io.File

fun main() {
    System.load("/usr/local/Cellar/opencv/4.1.0_2/share/java/opencv4/libopencv_java410.dylib")

    val folder = File("/Users/jagodevreede/git/openvalue/image-selector/images/Honeymoon")
    //val folder = File("/Volumes/Extern/2014-06-13_new_york/")
    //val folder = File("/Users/jagodevreede/git/openvalue/image-selector/images/small")
    val outputFolder = File("/Users/jagodevreede/git/openvalue/image-selector/images/output")
    outputFolder.mkdirs()
    outputFolder.listFiles().forEach { it.delete() }
    val reduceTo = 30

    Nima(folder).use {
        Runner(folder, outputFolder, it, reduceTo, 80.0).getResult()
    }

}

