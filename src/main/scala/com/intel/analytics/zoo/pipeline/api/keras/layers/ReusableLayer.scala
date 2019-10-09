package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, Table}

import scala.reflect.ClassTag

class ReusableLayer[T: ClassTag]
(delegate: KerasLayer[Tensor[T], Tensor[T], T], hasGradInput: Boolean)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T] {

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = if (hasGradInput) {
      labor.updateGradInput(input, gradOutput)
    } else {
      Activity.allocate[Tensor[T], T]()
    }
    gradInput
  }

  private var innerLabor: AbstractModule[Tensor[T], Tensor[T], T] = _
  private var outputShape: Shape = _

  override def computeOutputShape(inputShape: Shape): Shape = {
    outputShape = delegate.computeOutputShape(inputShape)
    outputShape
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = labor.updateOutput(input).clone()
    output
  }


  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    innerLabor = delegate.doBuild(inputShape)
    ReusableLayer.transformLabor(innerLabor)
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    innerLabor.parameters()
  }

  override def allowRebuilt(): Boolean = true

  override def skipDuplicateCheck(): Boolean = true

  def copy(name: String, isReplica: Boolean): KerasLayer[Tensor[T], Tensor[T], T] = {
    if (!isReplica) {
      this.setName(name)
    } else {
      new ReplicaLayer(ReusableLayer.transformLabor(innerLabor), outputShape, hasGradInput).setName(name)
    }
  }
}

class ReplicaLayer[T: ClassTag]
(delegateLabor: AbstractModule[Tensor[T], Tensor[T], T], delegateShape: Shape, hasGradInput: Boolean)
(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T] {

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = if (hasGradInput) {
      labor.updateGradInput(input, gradOutput)
    } else {
      Activity.allocate[Tensor[T], T]()
    }
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = delegateShape

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = labor.updateOutput(input).clone()
    output
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = delegateLabor

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  override def allowRebuilt(): Boolean = true

  override def skipDuplicateCheck(): Boolean = true
}

object ReusableLayer {
  def transformLabor[T: ClassTag](labor: AbstractModule[Tensor[T], Tensor[T], T])
                                 (implicit ev: TensorNumeric[T]): AbstractModule[Tensor[T], Tensor[T], T] = {
    labor match {
      case container: Container[Tensor[T], Tensor[T], T] =>
        new WrapperContainer[T](container)
      case _ => labor
    }
  }

}

class WrapperContainer[T: ClassTag](delegate: Container[Tensor[T], Tensor[T], T])(implicit ev: TensorNumeric[T])
  extends Container[Tensor[T], Tensor[T], T] {

  val moduleOutputs: Array[Activity] = Array.ofDim[Activity](delegate.modules.length)

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    moduleOutputs.zipWithIndex.foreach { case (module, i) => delegate.modules(i).output = moduleOutputs(i) }
    delegate.updateGradInput(input, gradOutput)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = delegate.updateOutput(input)
    delegate.modules.zipWithIndex.foreach { case (module, i) => moduleOutputs(i) = module.output match {
      case table: Table =>
        table.clone()
      case tensor: Tensor[T] => tensor.clone()
    }
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    moduleOutputs.zipWithIndex.foreach { case (module, i) => delegate.modules(i).output = moduleOutputs(i) }
    delegate.accGradParameters(input, gradOutput)
  }
}