package org.mrb.som

import scala.math.pow
import scala.math.log
import scala.math.max
import scala.math.min
import scala.math.exp

abstract class TrainingFunction(rows: Int, columns: Int, maxEpochs:Int, maxSteps: Int) {
  def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double;
  
  def lrLinear(timeStep: Int) : Double = 1.0 / timeStep
  def lrIoT(timeStep: Int) : Double = 1.0 - timeStep.toDouble/maxSteps
  def lrPowerSeries(timeStep: Int) : Double = pow(0.005, timeStep.toDouble/maxSteps)
  
  def fGaussian(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    val dx = x - cx
    val dy = y - cy
    val d2 = dx*dx + dy*dy
    val mapRadius = max(rows,columns) / 2
    val nc = max(1.0, mapRadius * (1.0 - (ts.toDouble/maxSteps)) / epoch.toDouble)
    exp( -d2 / (2*nc*nc) )
  }
  
  def fBubble(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    val dx = x - cx
    val dy = y - cy
    val d2 = (dx*dx) + (dy*dy)
    val mapRadius = max(rows,columns) / 2
    val nc = mapRadius * (1.0 - ts.toDouble/maxSteps)
    if (d2 <= nc) 1.0 else 0.0
  }
}