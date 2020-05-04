package util

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class UtilTest {
    @Test
    fun testFormat0() {
        Assertions.assertEquals("00:00:00.0", Util.toReadableTime(0))
    }

    @Test
    fun testFormat10000() {
        Assertions.assertEquals("00:00:10.0", Util.toReadableTime(10000))
    }
}