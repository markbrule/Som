package org.mrb.som

import scala.math._

/**
 * Classes for creating and managing a lattice (2-dimensional) of SOM Nodes
 * 
 * Constructor:
 *  (rows, cols) as Ints - the dimensions of the lattice
 *  (dim) as Int - the cardinality of the input vectors
 */
class SomLattice(var rows: Int, var cols: Int, var dim: Int) {
  var members = Array.ofDim[SomNode](rows,cols)
  
  def init() = {}
  
  /**
   * Return the SOM Node in the lattice that is closest to the provided input vector
   * Minimizes the "dist2" distance
   */
  def closestTo(ivec: Array[Double]) : SomNode = {
    members.flatten.minBy[Double](_.dist2(ivec))
  }
  
  /**
   * Apply learning function for one input vector
   */
  def applyLearningSingle(ivec: Array[Double], epoch: Int, ts: Int, 
      fn: (Double, Double, Double, Double, Int, Int) => Double) = {
    val center: SomNode = closestTo(ivec)
    members.flatten.foreach( _.updateWeights(ivec, center.x, center.y, epoch, ts, fn))
  }
  
  /**
   * Apply training cycle for specified number of epochs
   */
  def train(ivs: List[Array[Double]], epochs: Int, steps: Int, 
      fn: (Double, Double, Double, Double, Int, Int) => Double,
      progress: (SomLattice, Int, Int) => Unit) = {
    println("Beginning average E(QE) is " + averageQuantError(ivs))
    for {
      e <- 0 until epochs
      t <- 0 until steps
    } {
      if (t == 0) { 
        println("Average E(QE) beginning epoch " + (e+1) + " is " + averageQuantError(ivs))
      }
      progress(this, e, t)
      ivs.foreach( applyLearningSingle(_, e+1, t+1, fn ) )
    }
    println("Final average E(QE) is " + averageQuantError(ivs))
  }
  
  /**
   * Get the minimum and maximum distances (squared) to neighbors returned as a tuple (min,max)
   */
  def neighborDist2() : (Double, Double) = {
    members.flatten.map(_.neighborDist2())
      .reduce( (a:(Double,Double), b:(Double,Double)) => (min(a._1,b._1), max(a._2,b._2)))  
  }
  
  /**
   * Average quantization error for a set of vectors
   */
  def averageQuantError(ivec: List[Array[Double]]): Double = {
    (ivec zip ivec.map(closestTo(_))).map((x) => (x._2.dist2(x._1))).reduceLeft(_ + _) / ivec.length
  }
  
  /**
   * Provide a callback to each node in the U-Matrix
   * Note that this calls each U-Matrix node twice
   * fn parameter takes (nodex, nodey, node_distance)
   */
  def umatrixNodes(fn: (Double, Double, Double) => Unit) : Unit = {
    members.flatten.foreach((n) => {
      (n.ngbrs zip n.ngbrs.map(_.dist2(n.w))).foreach((d) => {
        val mx = (n.x+d._1.x)/2
        val my = (n.y+d._1.y)/2
        fn(mx, my, d._2) 
      })
    })
  }
  
  def show(neighbors: Boolean) : Unit = {
    var i, j = 0
    var s = ""
    for {
      i <- 0 until rows
      j <- 0 until cols
    } {
      println("Node at " + members(i)(j).toString + " has " + members(i)(j).neighborCount() + " neighbors")
      if (neighbors) members(i)(j).showNeighbors()
    }
  }

  def showDistances(ivec: Array[Double]) : Unit = {
    for {
      i <- 0 until rows
      j <- 0 until cols
    } {
      println("Distance to " + members(i)(j).toString + " is " + members(i)(j).dist2(ivec))
    }
  }
}
