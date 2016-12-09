package org.mrb.som

import scala.util.Random
import javax.json._

/**
 * Factory for creating different input vector instances
 */
object InputVectorFactory {
  def createRandom(dim: Int, count: Int): InputVectors = {
    var iv: InputVectors = new InputVectors()
    iv.dim = dim
    iv.fillRandom(dim, count, 0.0, 1.0)
    iv
  }
  
  def createClusters(dim: Int, base: List[Array[Double]], jitter: Double, count: Int): InputVectors = {
    var iv: InputVectors = new InputVectors()
    iv.dim = dim
    iv.fillClusters(dim, base, jitter, count)
    iv
  }

  def createFromConfig(cfg: JsonObject): InputVectors = {
    var iv: InputVectors = new InputVectors()
    iv.unserialize(cfg)
    iv
  }
}

/**
 * Input Vectors - for use in training or testing
 */
class InputVectors {
  var dim: Int = 0
  var ivecs: List[Array[Double]] = null
  
  /**
   * Fill with random values in the range provided
   */
  def fillRandom(d: Int, count: Int, min: Double, max: Double): InputVectors = {
    dim = d
    for { i <- 0 until count } {
      val x = Array.ofDim[Double](dim).map(_ => Random.nextDouble()*(max-min)+min)
      ivecs = if (ivecs == null) List(x) else ivecs :+ x
    }
    this
  }

  /**
   * Fill with random vectors clustered around a set of base vectors
   * 
   * @param d - dimension of the input vectors
   * @param base - array of base vectors, dimension should match d
   * @param jitter - maximum displacement in each dimension from the base vector
   * @param count - total number of vectors to generate
   * 
   * @return array of number of vectors of generated for each base vector
   */
  def fillClusters(d: Int, base: List[Array[Double]], jitter: Double, count: Int) : Array[Int] = {
    dim = d
    var r : Array[Int] = Array.ofDim[Int](base.length).map(_ => 0)
    for { i <- 0 until count } {
      val bi = Random.nextInt(base.length)
      r(bi) += 1
      val x = (base(bi) zip Array.ofDim[Double](dim).map(_ => Random.nextDouble()*jitter))
        .map((p) => p._1 + p._2)
      ivecs = if (ivecs == null) List(x) else ivecs :+ x
    }
    r
  }
  
  /**
 	 * Randomize one single dimension of the training vectors
 	 */
  def randomizeOneDimension(d: Int, min: Double, max: Double): InputVectors = {
    ivecs.map((a) => { 
      a(d) = min + Random.nextDouble()*(max-min)
      d
    })
    this
  }
  
  /**
   * Load from configuration
   */
  def unserialize(cfg: JsonObject) {
    dim = cfg.getInt("dim")
    ivecs = cfg.getJsonArray("vectors").toArray()
                   .map((a:Object) => a.asInstanceOf[JsonArray].toArray()
                   .map(_.asInstanceOf[JsonNumber].doubleValue())).toList
  }
  
  /**
   * Serialize to JSON object
   */
  def serialize(fact: JsonBuilderFactory = null) : JsonObject = {
    var factory = if (fact == null) Json.createBuilderFactory(null) else fact

    // training vectors
    val ivecBld = factory.createArrayBuilder()
    ivecs.foreach((v) => {
      val tf = factory.createArrayBuilder()
      v.foreach((vi) => tf.add(vi))
      ivecBld.add(tf.build())
    })
    
    factory.createObjectBuilder()
      .add("dim", dim)
      .add("vectors", ivecBld.build())
      .build()
  }
}