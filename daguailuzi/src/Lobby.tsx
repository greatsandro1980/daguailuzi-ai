/**
 * Lobby.tsx - 游戏大厅
 * 功能：填写昵称、选择队伍/座位、就绪等待、AI陪打、观战入口
 */

import { useState } from 'react';
import { useGame } from './GameContext';

const SEAT_CONFIG = [
  { seatIndex: 0, label: '红队 1号', team: 1 },
  { seatIndex: 2, label: '红队 2号', team: 1 },
  { seatIndex: 4, label: '红队 3号', team: 1 },
  { seatIndex: 1, label: '蓝队 1号', team: 2 },
  { seatIndex: 3, label: '蓝队 2号', team: 2 },
  { seatIndex: 5, label: '蓝队 3号', team: 2 },
];

export default function Lobby() {
  const {
    status, myNickname, mySeatIndex, isSpectator,
    room, setNickname, takeSeat, leaveSeat, toggleReady,
    addAI, removeAI, lastError, clearError
  } = useGame();

  const [nicknameInput, setNicknameInput] = useState('');
  const [nicknameSet, setNicknameSet] = useState(false);

  // AI 选择弹窗状态：null=关闭，number=目标座位index
  const [aiPickSeat, setAiPickSeat] = useState<number | null>(null);

  const handleSetNickname = () => {
    const name = nicknameInput.trim();
    if (!name) return;
    setNickname(name);
    setNicknameSet(true);
  };

  const handlePickAI = (aiId: string) => {
    if (aiPickSeat === null) return;
    addAI(aiPickSeat, aiId);
    setAiPickSeat(null);
  };

  const isMySeat = (si: number) => mySeatIndex === si;
  const isReady = mySeatIndex >= 0 && room?.seats[mySeatIndex]?.ready;

  const readyCount = room?.seats.filter(s => s?.ready).length ?? 0;
  const occupiedCount = room?.seats.filter(s => s?.occupied).length ?? 0;
  const usedAIIds = room?.usedAIIds ?? [];
  const aiCharacters = room?.aiCharacters ?? [];

  const renderSeat = (cfg: typeof SEAT_CONFIG[0]) => {
    const seatInfo = room?.seats[cfg.seatIndex];
    const occupied = seatInfo?.occupied;
    const occupiedByMe = isMySeat(cfg.seatIndex);
    const ready = seatInfo?.ready;
    const disconnected = seatInfo?.disconnected;
    const isAI = seatInfo?.isAI;
    const teamColor = cfg.team === 1 ? 'red' : 'blue';

    return (
      <div
        key={cfg.seatIndex}
        className={[
          'seat-card',
          occupiedByMe ? 'my-seat' : '',
          occupied && !occupiedByMe ? 'taken' : '',
          ready ? 'ready' : '',
          disconnected ? 'disconnected' : '',
          isAI ? 'ai-seat' : '',
        ].filter(Boolean).join(' ')}
      >
        <div className="seat-label">{cfg.label}</div>

        {occupied ? (
          <div className="seat-player">
            <span className="player-name">
              {seatInfo?.nickname}
              {isAI && <span className="ai-badge">AI</span>}
              {disconnected && !isAI && <span className="disconnect-badge">断线中</span>}
            </span>
            {ready && !disconnected && <span className="ready-badge">就绪</span>}
          </div>
        ) : (
          <div className="seat-empty">空位</div>
        )}

        {/* 按钮区 */}
        <div className="seat-actions">
          {/* 真人入座 */}
          {!occupied && nicknameSet && (
            <button
              className={`btn btn-sm btn-${teamColor}`}
              onClick={() => takeSeat(cfg.seatIndex)}
              disabled={mySeatIndex >= 0 && !occupiedByMe}
            >
              入座
            </button>
          )}

          {/* AI陪打 */}
          {!occupied && nicknameSet && (
            <button
              className="btn btn-sm btn-ai"
              onClick={() => setAiPickSeat(cfg.seatIndex)}
              disabled={mySeatIndex >= 0 && !occupiedByMe}
              title="让AI坐这个位置"
            >
              🤖 AI陪打
            </button>
          )}

          {/* 我的座位：离座 */}
          {occupiedByMe && (
            <button className="btn btn-sm btn-ghost" onClick={leaveSeat}>
              离座
            </button>
          )}

          {/* AI座位：移除 */}
          {isAI && !occupiedByMe && nicknameSet && (
            <button
              className="btn btn-sm btn-ghost"
              onClick={() => removeAI(cfg.seatIndex)}
            >
              撤除
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="lobby-container">
      {/* 连接状态 */}
      <div className={`conn-status ${status}`}>
        {status === 'connected' ? '● 已连接' : status === 'connecting' ? '⟳ 连接中...' : '✕ 已断开，重连中...'}
      </div>

      {/* 错误提示 */}
      {lastError && (
        <div className="lobby-error" onClick={clearError}>
          ⚠ {lastError} <span className="close-btn">✕</span>
        </div>
      )}

      <div className="lobby-header">
        <h1 className="lobby-title">大怪路子</h1>
        <p className="lobby-subtitle">六人斗牌 · 联机对战</p>
      </div>

      {/* 昵称设置 */}
      {!nicknameSet ? (
        <div className="nickname-panel">
          <h2>请输入您的昵称</h2>
          <div className="nickname-input-row">
            <input
              type="text"
              className="nickname-input"
              value={nicknameInput}
              onChange={e => setNicknameInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSetNickname()}
              placeholder="输入昵称（最多12字）"
              maxLength={12}
              autoFocus
            />
            <button className="btn btn-primary" onClick={handleSetNickname}>
              确认
            </button>
          </div>
        </div>
      ) : (
        <div className="nickname-confirmed">
          您的昵称：<strong>{myNickname}</strong>
          <button className="btn btn-sm btn-ghost" onClick={() => setNicknameSet(false)}>
            修改
          </button>
        </div>
      )}

      {/* 座位区 */}
      {nicknameSet && (
        <>
          <div className="seats-section">
            <div className="seats-grid">
              <div className="team-column team-red">
                <div className="team-header red">🔴 红队</div>
                {SEAT_CONFIG.filter(s => s.team === 1).map(renderSeat)}
              </div>
              <div className="team-column team-blue">
                <div className="team-header blue">🔵 蓝队</div>
                {SEAT_CONFIG.filter(s => s.team === 2).map(renderSeat)}
              </div>
            </div>
          </div>

          {/* 就绪按钮 & 状态 */}
          <div className="lobby-footer">
            <div className="ready-status">
              已就绪 {readyCount} / 6 &nbsp;·&nbsp; 在座 {occupiedCount} / 6
            </div>

            {mySeatIndex >= 0 && (
              <button
                className={`btn btn-ready ${isReady ? 'btn-ready-cancel' : 'btn-ready-confirm'}`}
                onClick={toggleReady}
              >
                {isReady ? '✓ 已就绪（点击取消）' : '点击就绪'}
              </button>
            )}

            {isSpectator && (
              <div className="spectator-hint">
                👁 所有座位均已占用，您将作为观战者观看游戏
              </div>
            )}

            {readyCount === 6 && occupiedCount === 6 && (
              <div className="all-ready-tip">所有人已就绪，游戏即将开始...</div>
            )}
          </div>

          {/* 房间消息 */}
          {room?.messages && room.messages.length > 0 && (
            <div className="lobby-messages">
              {room.messages.slice(-5).map(m => (
                <div key={m.id} className={`lobby-msg msg-${m.type}`}>{m.text}</div>
              ))}
            </div>
          )}
        </>
      )}

      {/* AI 角色选择弹窗 */}
      {aiPickSeat !== null && (
        <div className="ai-modal-overlay" onClick={() => setAiPickSeat(null)}>
          <div className="ai-modal" onClick={e => e.stopPropagation()}>
            <div className="ai-modal-title">
              🤖 选择 AI 陪打角色
              <span className="ai-modal-seat">
                {SEAT_CONFIG.find(s => s.seatIndex === aiPickSeat)?.label}
              </span>
            </div>
            <div className="ai-modal-grid">
              {aiCharacters.map(c => {
                const used = usedAIIds.includes(c.id);
                return (
                  <button
                    key={c.id}
                    className={`ai-char-btn ${used ? 'used' : ''}`}
                    onClick={() => !used && handlePickAI(c.id)}
                    disabled={used}
                    title={used ? '此角色已在其他座位' : `选择 ${c.name}`}
                  >
                    <span className="ai-char-emoji">{c.emoji}</span>
                    <span className="ai-char-name">{c.name}</span>
                    {used && <span className="ai-char-used">已入座</span>}
                  </button>
                );
              })}
            </div>
            <button className="btn btn-ghost ai-modal-cancel" onClick={() => setAiPickSeat(null)}>
              取消
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
