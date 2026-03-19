/**
 * MultiplayerApp.tsx
 * 应用根组件：根据游戏阶段决定显示大厅还是游戏界面
 */

import { GameProvider, useGame } from './GameContext';
import Lobby from './Lobby';
import OnlineGame from './OnlineGame';

function AppContent() {
  const { room, status } = useGame();

  // 正在连接
  if (status === 'connecting') {
    return (
      <div className="loading-screen">
        <div className="loading-spinner">⟳</div>
        <p>正在连接游戏服务器...</p>
      </div>
    );
  }

  // 断开连接
  if (status === 'disconnected') {
    return (
      <div className="loading-screen">
        <div className="loading-spinner error">✕</div>
        <p>服务器连接断开，正在重连...</p>
      </div>
    );
  }

  // 游戏进行中
  const gamePhase = room?.game?.phase;
  if (gamePhase && gamePhase !== 'lobby') {
    return <OnlineGame />;
  }

  // 大厅
  return <Lobby />;
}

export default function MultiplayerApp() {
  return (
    <GameProvider>
      <AppContent />
    </GameProvider>
  );
}
