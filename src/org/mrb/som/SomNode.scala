package org.mrb.som

import scala.math.min
import scala.math.max
import javax.json._
import java.util.UUID

/**
 * SomNode - class to implement a single Node in a SOM
 * 
 * Constructor:
 * 	(x, y) as Doubles to locate the node on a Cartesian Grid
 *  (dim) as Int identifying the cardinality of the input vectors
 *  f:(Int) => Array[Double] for initializing the weights on the node
 *
 */
class SomNode(var x: Double, var y: Double, var dim: Int, init: (Int) => Array[Double]) {
  var ngbrs: List[SomNode] = Nil
  var w: Array[Double] = init(dim)
  val id: UUID = UUID.randomUUID()
  
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
  
  /**
   * Find the minimum and maximum distances to the neighbors and return them as a tuple (min,max)
   */
  def neighborDist2() : (Double, Double) = {
    var d = ngbrs.map(_.dist2(w))
    val mx = d.reduce((a:Double, b:Double) => max(a,b))
    val mn = d.reduce((a:Double, b:Double) => min(a,b))
    (mn,mx)
  }
  
  /**
   * Create a JSON object for this SOM Node
   */
  def serialize(fact: JsonBuilderFactory = null): JsonObject = {
    var factory = if (fact == null) Json.createBuilderFactory(null) else fact
    val nFact = factory.createArrayBuilder()
    ngbrs.foreach((somNode) => nFact.add(somNode.id.toString()))
    val wFact = factory.createArrayBuilder()
    w.foreach((ws) => wFact.add(ws))
    
    factory.createObjectBuilder()
      .add("id", id.toString())
      .add("x", x)
      .add("y", y)
      .add("dim", dim)
      .add("ngbrs", nFact.build())
      .add("w", wFact.build())
      .build()
  }
  
  override def toString = { "loc = (" + "%.3f".format(x) + "," + "%.3f".format(y) + 
    ") weight = {" + w.map("%.6f".format(_)).reduceLeft(_ + "," + _) + "}" }
  
  def showNeighbors() = {
    println("Neighbors of " + this.toString)
    ngbrs.foreach ( (n: SomNode) => println("  " + n.toString) )
  }
}
