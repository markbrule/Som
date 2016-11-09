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
  def randomVecs: Int = 100
  val epochs = 1
  val its = 500
  val jitter = 0.1
  
  val start: Double = -20.0
  val end: Double = 20.0
  val step: Double = 2.0
  
  def main(args: Array[String]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    var img: Mat = new Mat(2*(columns+1)*radius, 2*(rows+1)*radius, CvType.CV_8UC3, new Scalar(0,0,0))
    var l: SomLattice = null
    var ivecs: List[Array[Double]] = null
    var v : Array[Double] = null
//    val theta: GaussianPower = new GaussianPower(rows, columns, epochs, its)
//    val theta: GaussianIoT = new GaussianIoT(rows, columns, epochs, its)
    val theta: GaussianLinear = new GaussianLinear(rows, columns, epochs, its)
    val progress: Progress = new Progress(0, 10, radius, img)    
/*
    for (t <- 1 until 202 by 100) {
      val f = eval_function(start, end, step, 1, t, theta.eval)
      println("Gaussian function, epoch = 1, time step = " + t)
      f.foreach((a) => println(a.map((x) => x._3.toString).reduceLeft(_ + "," + _)))
    }
    return
*/
//    l = new SomLatticeHex(rows, columns, dimension)
    l = new SomLatticeHex(rows, columns, dimension)
    l.init
    
    ivecs = List(
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
      val base = ivecs(scala.util.Random.nextInt(8))
      ivecs = ivecs ++ List(Array(max(0.0,min(1.0,scala.util.Random.nextDouble()*jitter+base(0))), 
                                  max(0.0,min(1.0,scala.util.Random.nextDouble()*jitter+base(1))), 
                                  max(0.0,min(1.0,scala.util.Random.nextDouble()*jitter+base(2)))))
    }

//    ivecs.foreach((x:Array[Double]) => println("(" + x(0).toString + "," + x(1).toString + "," + x(2).toString + ")"))
    println("Training for " + epochs + " epochs with " + its + " iterations per cycle")
    println("Network has " + (rows*columns) + " nodes")
    println("Total of " + ivecs.length + " input vectors")
    println("Using " + theta.getClass.getName)
    println("Jitter of " + jitter.toString)
    l.train(ivecs, epochs, its, theta.eval, progress.show)
    
    v = Array.fill[Double](dimension){scala.util.Random.nextDouble}
    println("Input vector = {" + v.map("%.3f".format(_)).reduceLeft(_ + "," + _) + "}")
    println("Closest node = " + l.closestTo(v).toString)
//    l.show(false)
    val mm = l.neighborDist2()
    println("Minimum distance = " + mm._1 + ", maximum = " + mm._2)
    
    progress.show(l, 1000000, 1000000)
    
    var umatrix: Mat = new Mat(2*(columns+1)*radius, 2*(rows+1)*radius, CvType.CV_8UC3, new Scalar(0,0,0))
    val yellow = new Scalar(255.0, 255.0, 0.0)
    l.members.flatten.foreach((n) =>
      (n.ngbrs zip n.ngbrs.map(_.neighborDist2())).foreach((d) => {
        val mid = new Point(radius+2*(n.x+d._1.x)*radius/2, radius+2*(n.y+d._1.y)*radius/2)
        val grey = 255.0 * d._2._2 / mm._2
        org.opencv.imgproc.Imgproc.circle(umatrix, mid, radius/2, new Scalar(grey, grey, grey), -1, 8, 0)
      }))
    l.members.flatten.foreach((n) => 
      org.opencv.imgproc.Imgproc.circle(umatrix, 
                                        new Point((2*n.x+1)*radius, (2*n.y+1)*radius), 
                                        (1.66*radius/2.0).toInt, 
                                        new Scalar(255.0*n.w(0), 255.0*n.w(1), 255.0*n.w(2)), 
                                        -1, 8, 0))

    org.opencv.imgcodecs.Imgcodecs.imwrite("/home/mark/umatrix.jpg", umatrix)
}
  
def eval_function(start: Double, end: Double, step: Double, epoch: Int, ts: Int, 
      fn: (Double, Double, Double, Double, Int, Int) => Double): Array[Array[(Double,Double,Double)]] = {
    val r = (start to end by step).toArray
    r.map(p => r.map(o => (p,o,fn(p,o,0.0,0.0,epoch,ts))))
  }
}


class Progress(epochStep: Int, iterationStep: Int, radius: Int, img: Mat) {
  val out_template: String = "/home/mark/test-%E-%T.jpg"
  def show(l: SomLattice, epoch: Int, step: Int): Unit =
  {
    if (epoch == -1 || (epochStep == 0) || ((epoch % epochStep) == 0)) {
      if (step == -1 || 
          (iterationStep == 0 && step == 0) || 
          (iterationStep == 1) || 
          ((iterationStep > 1) && ((step % iterationStep) == 0))) {
        l.members.flatten.foreach((n: SomNode) => org.opencv.imgproc.Imgproc.circle(img, new Point((2*(n.x)+1)*radius, (2*n.y+1)*radius), radius, color_node(n.w), -1, 8, 0))
        val path = out_template.replaceAll("%E", epoch.toString).replaceAll("%T", step.toString)
        org.opencv.imgcodecs.Imgcodecs.imwrite(path, img)
      }
    }
  }

  def color_node(w: Array[Double]): Scalar = {
    new Scalar(255.0*w(0), 255.0*w(1), 255.0*w(2))
  }
}
      
class GaussianLinear(rows: Int, columns: Int, maxEpochs:Int, maxSteps: Int) 
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fGaussian(x, y, cx, cy, epoch, ts) * lrLinear(ts)
  }
}

class GaussianIoT(rows: Int, columns: Int, maxEpochs: Int, maxSteps: Int)
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fGaussian(x, y, cx, cy, epoch, ts) * lrIoT(ts)
  }
}

class GaussianPower(rows: Int, columns: Int, maxEpochs: Int, maxSteps: Int)
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fGaussian(x, y, cx, cy, epoch, ts) * lrPowerSeries(ts)
  }
}

