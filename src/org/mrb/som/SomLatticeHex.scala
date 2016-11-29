package org.mrb.som

import scala.math._
import javax.json._

/**
 * Derived class: SomLatticeHex - a grid of hexagonally shaped nodes
 * Each node has 6 neighbors that are 1 unit away
 * 
 * (0,2)  (1,2)   (2,2)
 *    (0,1)   (1,1)   (2,1)
 * (0,0)  (1,0)   (2,0)
 */
//class SomLatticeHex(r: Int, c: Int, d: Int) extends SomLattice(r,c,d) {
class SomLatticeHex() extends SomLattice() {
  override def init() = {
    members = Array.ofDim[SomNode](rows,cols)
    if (config != null) 
      unserialize(config)
    else {  
      var i, j = 0
      val c60 = cos(Pi / 3)
      val s60 = sin(Pi / 3)
      
      for {
        i <- 0 until rows
        j <- 0 until cols
      } {
        if ((i % 2) == 0) {
          //members(i)(j) = new SomNode(j, i * s60, d, Array.ofDim[Double](_).map(_ => rand.nextDouble))
          members(i)(j) = new SomNode(j, i * s60, dim, Array.fill[Double](_){scala.util.Random.nextDouble()})
          if (j > 0) members(i)(j).addNeighbor(members(i)(j-1))
          if (i > 0 && j > 0) members(i)(j).addNeighbor(members(i-1)(j-1))
          if (i > 0) members(i)(j).addNeighbor(members(i-1)(j))
        } else {
          //members(i)(j) = new SomNode(j + c60, i *s60, d, Array.ofDim[Double](_).map(_ => rand.nextDouble))
          members(i)(j) = new SomNode(j + c60, i *s60, dim, Array.fill[Double](_){scala.util.Random.nextDouble()})
          if (j > 0) members(i)(j).addNeighbor(members(i)(j-1))
          if (i > 0) members(i)(j).addNeighbor(members(i-1)(j))
          if (i > 0 && j < cols-1) members(i)(j).addNeighbor(members(i-1)(j+1))
        }
      }
    }
  }
  
  override def latticeType() = "hex"
}