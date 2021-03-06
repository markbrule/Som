package org.mrb.som

import scala.math._
import javax.json._

/**
 * Derived class: SomLatticeRect - a rectangular grid of SOM Nodes
 * Four neighbors are 1 unit away (N, S, E, W)
 */
//class SomLatticeRect(r: Int, c: Int, d: Int) extends SomLattice(r,c,d) {
class SomLatticeRect() extends SomLattice() {
  override def init() = {    
    members = Array.ofDim[SomNode](rows,cols)
    var i, j = 0
    for {
      i <- 0 until rows
      j <- 0 until cols
    } {
      //members(i)(j) = new SomNode(j,i,d,Array.ofDim[Double](_).map(_ => rand.nextDouble))
      members(i)(j) = new SomNode(j,i,dim,Array.fill[Double](_){scala.util.Random.nextDouble()})
      if (i > 0) members(i)(j).addNeighbor(members(i-1)(j))
      if (j > 0) members(i)(j).addNeighbor(members(i)(j-1))
    }
  }

  override def latticeType() = "rect"
}
