package org.mrb.som

import org.mrb.som._
import scala.math._
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Point

object Test {
  def radius: Int = 25
  def rows: Int = 50
  def columns: Int = 50
  def dimension: Int = 3
  def out_template: String = "/home/mark/test-%E.jpg"
  def randomVecs: Int = 3000
  val epochs = 50
  val its = 50
  
  def main(args: Array[String]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    var img: Mat = new Mat(2*(columns+1)*radius, 2*(rows+1)*radius, CvType.CV_8UC3, new Scalar(0,0,0))

    var l : SomLattice = new SomLatticeHex(rows, columns, dimension)
    l.init
    
    var ivecs = List(
         Array(1.0, 0.0, 0.0),
         Array(0.0, 1.0, 0.0),
         Array(0.0, 0.0, 1.0),
         Array(1.0, 0.0, 1.0),
         Array(0.0, 1.0, 1.0),
         Array(1.0, 1.0, 0.0),
         Array(1.0, 1.0, 1.0),
         Array(0.0, 0.0, 0.0)
         )

    for {
      i <- 0 until randomVecs
    } {
      ivecs = ivecs ++ List(Array(scala.util.Random.nextDouble(), scala.util.Random.nextDouble(), scala.util.Random.nextDouble()))
    }

    l.members.flatten.foreach((n: SomNode) => org.opencv.imgproc.Imgproc.circle(img, new Point((2*(n.x)+1)*radius, (2*n.y+1)*radius), radius, color_node(n.w), -1, 8, 0))
    org.opencv.imgcodecs.Imgcodecs.imwrite(out_template.replaceAll("%E", "initial"), img)
    
    println("Training for " + epochs + " epochs with " + its + " iterations per cycle")
    println("Network has " + (rows*columns) + " nodes")
    println("Total of " + ivecs.length + " input vectors")
    println("Using gaussian_linear")
    l.train(ivecs, epochs, its, gaussian_linear, (l: SomLattice, epoch: Int) => {
      l.members.flatten.foreach((n: SomNode) => org.opencv.imgproc.Imgproc.circle(img, new Point((2*(n.x)+1)*radius, (2*n.y+1)*radius), radius, color_node(n.w), -1, 8, 0))
      val path = out_template.replaceAll("%E", epoch.toString)
      org.opencv.imgcodecs.Imgcodecs.imwrite(path, img)
    })
    
    var v : Array[Double] = Array.fill[Double](dimension){scala.util.Random.nextDouble}
    println("Input vector = {" + v.map("%.3f".format(_)).reduceLeft(_ + "," + _) + "}")
    println("Closest node = " + l.closestTo(v).toString)
    //l.show(false)

    l.members.flatten.foreach((n: SomNode) => org.opencv.imgproc.Imgproc.circle(img, new Point((2*(n.x)+1)*radius, (2*n.y+1)*radius), radius, color_node(n.w), -1, 8, 0))
    org.opencv.imgcodecs.Imgcodecs.imwrite(out_template.replaceAll("%E", "final"), img)
  }
  
  def bubble_linear(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    val dx = x - cx
    val dy = y - cy
    val d2 = (dx*dx) + (dy*dy)
    if (d2 <= 1.0) 1.0 / ts else 0.0
  }

  def gaussian_linear(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    val dx = x - cx
    val dy = y - cy
    val d2 = dx*dx + dy*dy
    val alpha = 1.0 / ts
    val sigma = 1.0 / (epoch * ts)
    val v = alpha * exp(- sqrt(d2) / (2 * sigma * sigma))
    v
  }
  
  def color_node(w: Array[Double]): Scalar = {
    new Scalar(255.0*w(0), 255.0*w(1), 255.0*w(2))
  }
}