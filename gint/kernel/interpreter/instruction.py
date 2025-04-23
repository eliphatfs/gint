from ..platforms.platform import PlatformIRBuilder


class Instruction:
    
    def emit(self, builder: PlatformIRBuilder):
        raise NotImplementedError
