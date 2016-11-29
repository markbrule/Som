package org.mrb.som

import scala.math._
import javax.json._

object SomLatticeFactory {
  def createLattice(t: String, rows: Int, cols: Int, dim: Int) : SomLattice = {
    var lat: SomLattice = if (t == "rect") new SomLatticeRect() else new SomLatticeHex()
    lat.rows = rows
    lat.cols = cols
    lat.dim = dim
    lat.config = null
    lat.init
    lat
  }
  
  def createLattice(cfg: JsonObject) : SomLattice = {
    val t: String = cfg.getString("type")
    var lat: SomLattice = if (t == "rect") new SomLatticeRect() else new SomLatticeHex()
    lat.rows = cfg.getInt("rows")
    lat.cols = cfg.getInt("cols")
    lat.config = cfg
    lat.init
    lat
  }
}

/**
 * Classes for creating and managing a lattice (2-dimensional) of SOM Nodes
 * 
 * Constructor:
 *  (rows, cols) as Ints - the dimensions of the lattice
 *  (dim) as Int - the cardinality of the input vectors
 */
abstract class SomLattice() {

  var rows: Int = 0
  var cols: Int = 0
  var dim: Int = 0
  var config: JsonObject = null
  var members = Array.ofDim[SomNode](rows,cols)
  
  def init()
  def latticeType(): String
  
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
  
  /**
   * Pick an (x,y) that represents a given input vector
   * Find the BMU then offset it toward a weighted average of the neighbors
   */
  def selectRepPt(iv: Array[Double]): (Double, Double) = {
    val bmu: SomNode = closestTo(iv)
    val dist2Bmu = bmu.dist2(iv)
    var xy = (bmu.x, bmu.y)
// TODO   if (dist2Bmu > 0) {
      val jj: List[(Double,Double)] = bmu.ngbrs.map((n) => (dist2Bmu/n.dist2(iv)*(n.x-bmu.x), dist2Bmu/n.dist2(iv)*(n.y-bmu.y)))
      val xxy: (Double,Double) = jj.reduce((xy1,xy2) => (xy1._1+xy2._1, xy1._2+xy2._2))
      xy = (bmu.x+xxy._1/bmu.ngbrs.length, bmu.y+xxy._2/bmu.ngbrs.length)
//    }
    xy
  }
  
  /**
   * Create a JSON object for the lattice
   */
  def serialize(fact: JsonBuilderFactory = null) : JsonObject = {
    var factory = if (fact == null) Json.createBuilderFactory(null) else fact
    // members factory
    val mFact = factory.createArrayBuilder()
    members.flatten.foreach((somNode) => mFact.add(somNode.serialize(factory)))
    val sFact = factory.createArrayBuilder()
    for { r <- 0 until rows
          c <- 0 until cols
    } { 
        sFact.add(factory.createObjectBuilder()
                  .add("row",r)
                  .add("column",c)
                  .add("node",members(r)(c).id.toString)
                  .build())
    }
    factory.createObjectBuilder()
      .add("type", latticeType)
      .add("rows", rows)
      .add("cols", cols)
      .add("members", mFact.build())
      .add("structure", sFact.build())
      .build()
  }
  
  /**
   * Recreate the lattice from a JSON object
   */
  def unserialize(c: JsonObject) = {
    import scala.collection.mutable.HashMap
    var nodes: HashMap[String,SomNode] = new HashMap[String,SomNode]()
    var neighbors: HashMap[String,Array[String]] = new HashMap[String,Array[String]]
    def createWeights(o: JsonObject): Array[Double] = o.getJsonArray("w").toArray().map((v:Object) => v.toString().toDouble)
    c.getJsonArray("members")
      .toArray()
      .map((o:Object) => o.asInstanceOf[JsonObject])
      .foreach((o:JsonObject) => {
        val sn = new SomNode(o.getJsonNumber("x").doubleValue(), o.getJsonNumber("y").doubleValue(), o.getInt("dim"), (_) => createWeights(o))
        nodes.put(o.getString("id"), sn)
        neighbors.put(o.getString("id"), o.getJsonArray("ngbrs").toArray().map(_.asInstanceOf[JsonString].getString()))
      })
    neighbors.foreach((p:(String,Array[String])) => p._2.foreach((t:String) => nodes.getOrElse(p._1,null).addNeighbor(nodes.getOrElse(t,null))))
    c.getJsonArray("structure")
      .toArray()
      .map((o:Object) => o.asInstanceOf[JsonObject])
      .foreach((jo:JsonObject) => members(jo.getInt("row"))(jo.getInt("column")) = nodes.getOrElse(jo.getString("node"),null))
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
