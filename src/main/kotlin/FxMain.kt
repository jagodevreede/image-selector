import javafx.application.Application
import javafx.application.Platform
import javafx.event.EventHandler
import javafx.geometry.Insets
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.effect.DropShadow
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.input.MouseEvent
import javafx.scene.layout.ColumnConstraints
import javafx.scene.layout.GridPane
import javafx.scene.paint.Color
import javafx.stage.DirectoryChooser
import javafx.stage.Screen
import javafx.stage.Stage
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import util.CV
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.time.format.DateTimeFormatter
import java.util.*
import kotlin.concurrent.thread

class FxMain: Application() {
    var result = mutableListOf<FileStats>()
    lateinit var runner: Runner
    lateinit var root: GridPane
    lateinit var labelSelectedDirectory: Label
    lateinit var primaryStage: Stage
    private var gridPane: ScrollPane? = null
    private val properties = Properties()
    private val propertiesFile = File("settings.properties")

    lateinit var selectedFolder: String
    val outputFolder = File("/Users/jagodevreede/git/openvalue/image-selector/images/output")
    var reduceTo = 30
    var similarThreshold = 80.0

    val imageCache = mutableMapOf<File, Image>()

    val removedImages = mutableSetOf<FileStats>()

    private val progressIndicator = ProgressIndicator(0.0)


    override fun start(primaryStage: Stage) {
        this.primaryStage = primaryStage

        if (propertiesFile.exists()) {
            properties.load(FileInputStream(propertiesFile))
        }

        primaryStage.isMaximized = true
        primaryStage.title = "Image selector"

        outputFolder.mkdirs()
        outputFolder.listFiles().forEach { it.delete() }

        root = buildRootPane()
        progressIndicator.minWidth = 500.0
        progressIndicator.minHeight = 300.0

        val buttonPane = buildButtonPane()
        root.add(buttonPane, 1, 0)

        val scene = Scene(root, 800.0, 800.0)
        primaryStage.scene = scene

        primaryStage.show()
        primaryStage.toFront()

        drawResult()
    }

    private fun drawResult() {
        Platform.runLater {
            if (gridPane != null) {
                root.children.remove(gridPane)
            }
            root.add(progressIndicator, 0, 0)
            progressIndicator.progress = 0.0
        }
        thread {
            val visualBounds = Screen.getPrimary().visualBounds
            gridPane = loadGrid(result, visualBounds.width)
            Platform.runLater {
                root.children.remove(progressIndicator)
                root.add(gridPane, 0, 0)
            }
        }
    }

    private fun buildButtonPane(): GridPane {
        val buttonPane = GridPane()

        buttonPane.padding = Insets(20.0)
        buttonPane.hgap = 20.0
        buttonPane.vgap = 15.0

        labelSelectedDirectory = Label("No Directory selected")

        val btnOpenDirectoryChooser = Button()
        btnOpenDirectoryChooser.text = "Open Directory"
        btnOpenDirectoryChooser.onAction = EventHandler {
            val directoryChooser = DirectoryChooser()
            directoryChooser.initialDirectory = File(properties.getProperty("last_dir", System.getProperty("user.home")))
            val selectedDirectory = directoryChooser.showDialog(primaryStage)

            if (selectedDirectory == null) {
                labelSelectedDirectory.text = "No Directory selected"
            } else {
                labelSelectedDirectory.text = selectedDirectory.absolutePath
                properties.put("last_dir", selectedDirectory.absolutePath)
                selectedFolder = selectedDirectory.absolutePath
            }
        }

        buttonPane.add(labelSelectedDirectory, 0, 0)
        buttonPane.add(btnOpenDirectoryChooser, 0, 1)

        val reduceToInput = TextField(reduceTo.toString())
        reduceToInput.maxWidth = 50.0
        reduceToInput.textProperty().addListener { _, oldValue, newValue ->
            if (!newValue.matches("\\d{0,3}?".toRegex())) {
                reduceToInput.text = oldValue
            }
            if (newValue == "") {
                reduceTo = 0
            } else {
                reduceTo = newValue.toInt()
            }
        }

        val gp = GridPane()
        gp.add(Label("Reduce to: "), 0, 0)
        gp.add(reduceToInput, 1, 0)
        buttonPane.add(gp, 0, 2)

        val slider = Slider()
        slider.min = 0.0
        slider.max = 100.0
        slider.value = similarThreshold
        slider.isShowTickLabels = true
        slider.isShowTickMarks = true

        slider.blockIncrement = 10.0

        slider.valueProperty().addListener { _, _, newValue -> similarThreshold = newValue.toDouble() }
        buttonPane.add(Label("Similar threshold: "), 0, 3)
        buttonPane.add(slider, 0, 4)

        val refresh = Button("Load/Refresh")
        refresh.onMouseClicked = EventHandler {
            onRefreshClick()
        }
        buttonPane.add(refresh, 0, 5)

        val remove = Button("Remove")
        remove.onMouseClicked = EventHandler {
            onRemoveClick(it)
        }
        buttonPane.add(remove, 0, 6)
        val doneButton = Button("Done")
        doneButton.onMouseClicked = EventHandler {
            onDoneClick(it)
        }
        buttonPane.add(doneButton, 0, 7)
        return buttonPane
    }

    private fun onRefreshClick() {
        runner = Runner(File(selectedFolder), outputFolder, Nima(File(selectedFolder)), reduceTo, similarThreshold / 100)
        result = runner.getResult()

        drawResult()
    }

    private fun buildRootPane(): GridPane {
        val root = GridPane()

        val colConst1 = ColumnConstraints()
        colConst1.percentWidth = 80.0
        root.columnConstraints.add(colConst1)

        val colConst2 = ColumnConstraints()
        colConst2.percentWidth = 20.0
        root.columnConstraints.add(colConst2)
        return root
    }

    private fun onDoneClick(e: MouseEvent) {
        properties.store(FileOutputStream(propertiesFile), "")
        runner.copyToOutput()
        Platform.exit()
    }

    private fun onRemoveClick(e: MouseEvent) {
        removedImages.addAll(result.filter { it.ignored })
        result.addAll(removedImages)
        result = runner.updateResults(result)
        drawResult()
    }

    fun loadGrid(result: List<FileStats>, width: Double): ScrollPane {
        val root = GridPane()

        val numCols = 3
        for (i in 0 until numCols) {
            val colConst = ColumnConstraints()
            colConst.percentWidth = 100.0 / numCols
            root.columnConstraints.add(colConst)
        }

        root.padding = Insets(20.0)
        root.hgap = 25.0
        root.vgap = 15.0

        val newWidth = ((width * 0.8 /3) - root.padding.right - root.vgap *2)

        result.forEachIndexed { index, fileStat ->

            val iv2 = ImageView()
            iv2.id = fileStat.file.absolutePath

            iv2.image = imageCache.getOrPut(fileStat.file, {
                val img = CV.loadMat(fileStat.file)

                val resizeimage = Mat();
                val sz = Size(newWidth, img.height() / (img.width() / newWidth))

                Imgproc.resize(img, resizeimage, sz)
                CV.matToJavaFxImage(resizeimage)
            })

            iv2.onMouseClicked = EventHandler {
                val imageView = it.target as ImageView
                onImageClicked(imageView)
            }

            Tooltip.install(iv2, Tooltip("""${fileStat.file.name}
                |Rating: %.2f
                |Sharpness: ${fileStat.sharpness}
                |Faces: ${fileStat.faces}
                |Date taken: ${fileStat.dateTaken.format(DateTimeFormatter.ISO_DATE_TIME)}
                |Resolution: ${fileStat.size}
            """.trimMargin().format(fileStat.rating)))


            root.add(iv2,  index % 3, index / 3)

            Platform.runLater { progressIndicator.progress = index.toDouble() / result.count()}
        }

        return ScrollPane(root)
    }

    private fun onImageClicked(imageView: ImageView) {
        val item = result.find { imageView.id == it.file.absolutePath }!!
        if (item.ignored) {
            imageView.effect = null
        } else {
            imageView.effect = DropShadow(25.0, Color.RED)
        }
        val newItem = item.copy(ignored = !item.ignored)
        result.remove(item)
        result.add(newItem)
        result.sortBy { it.file.name }
    }

    override fun stop() {
        properties.store(FileOutputStream(propertiesFile), "")
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            System.load("/usr/local/Cellar/opencv/4.1.0_2/share/java/opencv4/libopencv_java410.dylib")

            launch(FxMain::class.java)
        }
    }
}