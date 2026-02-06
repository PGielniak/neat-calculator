package com.example.hellom8

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class TimestampOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var timestampText = ""
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        style = Paint.Style.FILL
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }
    
    private val backgroundPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }
    
    private val textBounds = Rect()

    fun updateTimestamp(timestamp: Long) {
        timestampText = timestamp.toString()
        invalidate()
    }
    
    private fun formatTimestamp(timestamp: Long): String {
        return timestamp.toString()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        if (timestampText.isNotEmpty()) {
            textPaint.getTextBounds(timestampText, 0, timestampText.length, textBounds)
            
            val padding = 20f
            val x = width - textBounds.width() - padding * 2
            val y = height - padding * 2
            
            // Draw background
            canvas.drawRect(
                x - padding,
                y - textBounds.height() - padding,
                width.toFloat() - padding,
                y + padding,
                backgroundPaint
            )
            
            // Draw text
            canvas.drawText(timestampText, x, y, textPaint)
        }
    }
}
