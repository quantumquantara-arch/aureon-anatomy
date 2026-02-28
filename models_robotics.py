class RoboticConnection(BaseModel):
    id: str
    name: str
    robot_type: str  # "industrial_arm", "humanoid", "amr", "custom", etc.
    interface_protocol: str  # "ROS2", "OPC_UA", "EtherCAT", etc.
    connection_status: Literal["connected", "disconnected", "error"]
    last_heartbeat: datetime
    dgk_signature: str
    user_id: str

class PhysicalActionLog(BaseModel):
    id: str
    robotic_connection_id: str
    command: dict
    execution_status: Literal["planned", "executing", "completed", "failed", "rejected"]
    dgk_certificate_id: str
    timestamp: datetime
    sensor_snapshot: dict  # fused perception data
    user_id: str
