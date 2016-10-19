package org.mrb.som

/**
 * SomNode - class to implement a single Node in a SOM
 * 
 * Constructor:
 * 	(x, y) as Doubles to locate the node on a Cartesion Grid
 *  (dim) as Int identifying the cardinality of the input vectors
 *  f:(Int) => Array[Double] for initializing the weights on the node
 *
 */
class SomNode(var x: Double, var y: Double, var dim: Int, init: (Int) => Array[Double]) {
  var ngbrs: List[SomNode] = Nil
  var w: Array[Double] = init(dim)
  
  /**
   * Add a relationship to a neighboring node. Neighbors that are 1 unit away should be related
   * This method creates the reciprocal relationship from the other node
   */
  def addNeighbor(that: SomNode) : Unit = {
    if (ngbrs != Nil && ngbrs.contains(that))
      return
    ngbrs = if (ngbrs == Nil) List(that) else ngbrs ++ List(that)
    that.ngbrs = if (that.ngbrs == Nil) List(this) else that.ngbrs ++ List(this)
  }
  
  /**
   * Calculate the distance^2 to a weight vector
   */
  def dist2(that: Array[Double]) : Double = {
    (w zip that).map((x) => (x._1 - x._2)*(x._1 - x._2)).reduceLeft(_ + _);
  }
  
  /**
   * Return the number of neighbors to this node
   */
  def neighborCount() : Int = {
     return ngbrs.length
  }
 
  /**
   * Apply learning function
   */
  def updateWeights(ivec: Array[Double], cx: Double, cy: Double, epoch: Int, ts: Int, 
      fn: (Double, Double, Double, Double, Int, Int) => Double) = {
    val h = fn(x, y, cx, cy, epoch, ts)
    for ( i <- 0 until dim ) { w.update(i, w(i) + h*(ivec(i) - w(i))) }
  }
  
  override def toString = { "loc = (" + "%.3f".format(x) + "," + "%.3f".format(y) + 
    ") weight = {" + w.map("%.3f".format(_)).reduceLeft(_ + "," + _) + "}" }
  
  def showNeighbors() = {
    println("Neighbors of " + this.toString)
    ngbrs.foreach ( (n: SomNode) => println("  " + n.toString) )
  }
}
